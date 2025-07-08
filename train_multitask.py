import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, DebertaV2Model, DebertaV2Tokenizer
from tqdm.auto import tqdm
import numpy as np
import logging
from collections import defaultdict
from torch.amp import autocast, GradScaler

# Configura o logger
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

# ===== CONFIG =====
DATA_PATH = "/home/yansoares/mlp_finetuning_embedding/data/downstream"
MODEL_NAME = "microsoft/deberta-v3-base" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 50
MAX_LEN = 512 
LEARNING_RATE = 2e-5
EARLY_STOPPING_PATIENCE = 5
SAVE_DIR = "/home/yansoares/mlp_finetuning_embedding_models/review"
os.makedirs(SAVE_DIR, exist_ok=True)

# Funções de carregar dataset (mantidas como estavam)
def load_binary_dataset(pos_path, neg_path):
    with open(pos_path, "r", encoding="latin1") as f: pos = [line.strip() for line in f]
    with open(neg_path, "r", encoding="latin1") as f: neg = [line.strip() for line in f]
    return pos + neg, [1]*len(pos) + [0]*len(neg)

def load_sst_dataset(path):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                texts.append(parts[0])
                labels.append(int(parts[1]))
    return texts, labels

def load_trec_dataset(path):
    texts, labels_str = [], []
    with open(path, "r", encoding="latin1") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                labels_str.append(parts[0].split(":")[0])
                texts.append(parts[1])
    label_to_id = {label: idx for idx, label in enumerate(sorted(set(labels_str)))}
    return texts, [label_to_id[l] for l in labels_str]

def load_mrpc_dataset(path):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 5:
                texts.append(parts[3] + " [SEP] " + parts[4])
                labels.append(int(parts[0]))
    return texts, labels

# ===================================================================
# ===== ARQUITETURA DO MODELO (MODIFICADO PARA MULTITASK) =====
# ===================================================================

class LearnablePooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention_scorer = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).float()
        scores = self.attention_scorer(x)
        
        # CORREÇÃO: Em vez de -1e9, use o menor valor que o dtype atual suporta.
        # Isso evita o overflow quando em precisão mista (float16).
        scores[mask_expanded == 0] = torch.finfo(scores.dtype).min
        
        weights = F.softmax(scores, dim=1)
        return (x * weights).sum(dim=1)

class LayerAggregator(nn.Module):
    def __init__(self, num_layers=12):
        super().__init__()
        self.combiner = nn.Linear(in_features=num_layers, out_features=1, bias=False)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.combiner(x)
        return x.squeeze(-1)

class CustomClassifier(nn.Module):
    def __init__(self, model_name: str, task_config: dict):
        super().__init__()
        self.base_model = DebertaV2Model.from_pretrained(model_name, output_hidden_states=True)
        hidden_size = self.base_model.config.hidden_size
        num_layers = self.base_model.config.num_hidden_layers
        self.layer_poolers = nn.ModuleList([LearnablePooling(hidden_size) for _ in range(num_layers)])
        self.layer_aggregator = LayerAggregator(num_layers)
        self.classifiers = nn.ModuleDict({
            task_name: nn.Linear(hidden_size, num_classes)
            for task_name, num_classes in task_config.items()
        })
        self.id_to_task_name = {i: name for i, name in enumerate(task_config.keys())}


    def forward(self, input_ids, attention_mask, task_ids):
        # O método forward de treinamento continua o mesmo, pois é necessário para o treino.
        final_embedding = self.get_embedding_base(input_ids, attention_mask)
        
        # Roteamento para a cabeça de classificação correta...
        max_num_classes = max(head.out_features for head in self.classifiers.values())

        # CORREÇÃO: Crie o tensor `all_logits` com o mesmo dtype do embedding.
        # `final_embedding.dtype` será `torch.float16` dentro do autocast.
        all_logits = torch.zeros(
            final_embedding.size(0), 
            max_num_classes, 
            device=final_embedding.device, 
            dtype=final_embedding.dtype
        )

        for i, task_name in self.id_to_task_name.items():
            task_mask = (task_ids == i)
            if task_mask.any():
                task_embedding = final_embedding[task_mask]
                task_logits = self.classifiers[task_name](task_embedding)
                num_classes_for_task = task_logits.shape[1]
                all_logits[task_mask, :num_classes_for_task] = task_logits
        
        return all_logits

    # NOVO: Método base para extrair a parte compartilhada
    def get_embedding_base(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states[-self.base_model.config.num_hidden_layers:]
        
        layer_outputs = [self.layer_poolers[i](hidden_states[i], attention_mask) for i in range(len(hidden_states))]
        layer_stack = torch.stack(layer_outputs, dim=1)
        
        final_embedding = self.layer_aggregator(layer_stack)
        return final_embedding

    # NOVO: Método dedicado para inferência de embeddings
    def get_embedding(self, input_ids, attention_mask):
        """
        Gera os embeddings generalistas para um batch de textos.
        NÃO precisa de task_ids.
        """
        self.eval() # Garante que o modelo está em modo de avaliação (desativa dropout, etc.)
        with torch.no_grad(): # Desativa o cálculo de gradientes para economizar memória e tempo
            embedding = self.get_embedding_base(input_ids, attention_mask)
        return embedding

# NOVO: Dataset que lida com múltiplas tarefas
class MultitaskDataset(Dataset):
    def __init__(self, tasks_data: dict, tokenizer, max_len):
        """
        Args:
            tasks_data (dict): Dicionário onde chaves são nomes de tarefas e valores
                               são tuplas (texts, labels).
            tokenizer: O tokenizador do Hugging Face.
            max_len (int): Comprimento máximo da sequência.
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        self.task_to_id = {task_name: i for i, task_name in enumerate(tasks_data.keys())}

        # Constrói uma lista única de amostras (texto, rótulo, id_da_tarefa)
        for task_name, (texts, labels) in tasks_data.items():
            task_id = self.task_to_id[task_name]
            for text, label in zip(texts, labels):
                self.samples.append({
                    "text": text,
                    "label": label,
                    "task_id": task_id
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        sample = self.samples[item]
        text = str(sample["text"])
        label = sample["label"]
        task_id = sample["task_id"]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'task_ids': torch.tensor(task_id, dtype=torch.long) # NOVO: Retorna o ID da tarefa
        }

# ===================================================================
# ===== MAIN TRAINING SCRIPT (MODIFICADO PARA MULTITASK) =====
# ===================================================================

def main():
    print(f"Usando dispositivo: {DEVICE}")
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)

    datasets_raw = {
        "MR": load_binary_dataset(os.path.join(DATA_PATH, "MR", "rt-polarity.pos"), os.path.join(DATA_PATH, "MR", "rt-polarity.neg")),
        "CR": load_binary_dataset(os.path.join(DATA_PATH, "CR", "custrev.pos"), os.path.join(DATA_PATH, "CR", "custrev.neg")),
        "MPQA": load_binary_dataset(os.path.join(DATA_PATH, "MPQA", "mpqa.pos"), os.path.join(DATA_PATH, "MPQA", "mpqa.neg")),
        "SUBJ": load_binary_dataset(os.path.join(DATA_PATH, "SUBJ", "subj.subjective"), os.path.join(DATA_PATH, "SUBJ", "subj.objective")),
        "SST2": load_sst_dataset(os.path.join(DATA_PATH, "SST", "binary", "sentiment-train")),
        "TREC": load_trec_dataset(os.path.join(DATA_PATH, "TREC", "train_5500.label")),
        "MRPC": load_mrpc_dataset(os.path.join(DATA_PATH, "MRPC", "msr_paraphrase_train.txt"))
    }

    for held_out_task in datasets_raw:
        print(f"\n{'='*25}\n🔁 Treinando com exclusão de: {held_out_task}\n{'='*25}")

        # MODIFICADO: Prepara os dados para o formato multitask
        train_tasks_data = {name: data for name, data in datasets_raw.items() if name != held_out_task}
        
        # MODIFICADO: Cria o dicionário de configuração para o modelo
        train_tasks_config = {name: len(set(labels)) for name, (texts, labels) in train_tasks_data.items()}
        print(f"Tarefas de treinamento: {train_tasks_config}")

        # MODIFICADO: Usa o MultitaskDataset
        full_dataset = MultitaskDataset(train_tasks_data, tokenizer, MAX_LEN)
        
        # Separando treino e validação
        val_size = int(len(full_dataset) * 0.2)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)

        # MODIFICADO: Instancia o modelo com a configuração das tarefas
        model = CustomClassifier(model_name=MODEL_NAME, task_config=train_tasks_config).to(DEVICE)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        criterion = nn.CrossEntropyLoss().to(DEVICE)
        
        # PASSO A: Inicialize o GradScaler.
        scaler = GradScaler()

        best_avg_val_accuracy = 0.0
        epochs_no_improve = 0
        
        id_to_task_name_map = model.id_to_task_name
        
        for epoch in range(EPOCHS):
            print(f'\n--- Epoch {epoch + 1}/{EPOCHS} ---')
            model.train()
            total_train_loss = 0
            
            progress_bar = tqdm(train_loader, desc="Training", leave=False)
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                task_ids = batch['task_ids'].to(DEVICE)

                optimizer.zero_grad()
                
                # PASSO B: O contexto autocast envolve a chamada do modelo.
                with autocast(device_type='cuda'):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask, task_ids=task_ids)
                    
                    # Lógica do loop de tarefas
                    unique_task_ids = torch.unique(task_ids)
                    num_unique_tasks = len(unique_task_ids)
                    total_loss_for_logging = 0

                    for i, task_id_val in enumerate(unique_task_ids):
                        task_mask = (task_ids == task_id_val)
                        
                        # O cálculo da perda da tarefa também deve estar dentro do autocast
                        task_logits_full = logits[task_mask]
                        task_labels = labels[task_mask]
                        
                        num_classes_for_task = train_tasks_config[id_to_task_name_map[task_id_val.item()]]
                        task_logits_sliced = task_logits_full[:, :num_classes_for_task]
                        
                        loss_per_task = criterion(task_logits_sliced, task_labels)
                        
                        total_loss_for_logging += loss_per_task.item()
                        
                        # PASSO C: Escale a perda da TAREFA ATUAL e faça o backward.
                        # A lógica de `retain_graph` é mantida.
                        retain_graph_needed = (i < num_unique_tasks - 1)
                        scaler.scale(loss_per_task).backward(retain_graph=retain_graph_needed)

                # PASSO D: Fora do loop de tarefas, mas dentro do loop do batch.
                # Otimize os pesos usando os gradientes acumulados de todas as tarefas.
                scaler.step(optimizer)

                # PASSO E: Atualize o scaler para o próximo batch.
                scaler.update()

                scheduler.step()
                
                total_train_loss += total_loss_for_logging
                progress_bar.set_postfix({'loss': total_loss_for_logging})
            
            avg_train_loss = total_train_loss / len(train_loader)
            print(f'Average training loss: {avg_train_loss:.4f}')

            # O loop de validação não precisa de scaler, pois não há retropropagação.
            # Ele pode se beneficiar do `autocast` para ser mais rápido e usar menos memória.
            model.eval()
            correct_predictions = defaultdict(int)
            total_predictions = defaultdict(int)
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation", leave=False):
                    input_ids = batch['input_ids'].to(DEVICE)
                    attention_mask = batch['attention_mask'].to(DEVICE)
                    labels = batch['labels'].to(DEVICE)
                    task_ids = batch['task_ids'].to(DEVICE)

                    with autocast(device_type='cuda'):
                        logits = model(input_ids=input_ids, attention_mask=attention_mask, task_ids=task_ids)
                    
                    # Calcula acurácia para cada tarefa presente no batch
                    for task_id_val in torch.unique(task_ids):
                        task_mask = (task_ids == task_id_val)
                        task_name = id_to_task_name_map[task_id_val.item()]
                        num_classes_for_task = train_tasks_config[task_name]
                        
                        task_logits = logits[task_mask][:, :num_classes_for_task]
                        task_labels = labels[task_mask]
                        
                        _, preds = torch.max(task_logits, dim=1)
                        
                        correct_predictions[task_name] += torch.sum(preds == task_labels).item()
                        total_predictions[task_name] += task_labels.size(0)

            # Calcula e imprime a acurácia por tarefa e a média
            task_accuracies = {name: correct / total if total > 0 else 0 
                               for name, correct, total in 
                               ((n, correct_predictions[n], total_predictions[n]) for n in train_tasks_config.keys())}

            for name, acc in task_accuracies.items():
                print(f'  - Val Accuracy for {name}: {acc:.4f}')
            
            avg_val_accuracy = np.mean(list(task_accuracies.values()))
            print(f'Average Validation Accuracy: {avg_val_accuracy:.4f}')
            
            # Lógica do Early Stopping baseada na acurácia média
            if avg_val_accuracy > best_avg_val_accuracy:
                best_avg_val_accuracy = avg_val_accuracy
                epochs_no_improve = 0
                output_path = os.path.join(SAVE_DIR, f"model_exclude_{held_out_task}.bin")
                torch.save(model.state_dict(), output_path)
                print(f"✅ Melhoria na acurácia média! Melhor modelo salvo com {best_avg_val_accuracy:.4f} em: {output_path}")
            else:
                epochs_no_improve += 1
                print(f"Sem melhoria na acurácia média por {epochs_no_improve} épocas.")

            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"\n✋ Early stopping acionado após {EARLY_STOPPING_PATIENCE} épocas sem melhoria.")
                print(f"Melhor acurácia média de validação obtida: {best_avg_val_accuracy:.4f}")
                break

if __name__ == "__main__":
    main()