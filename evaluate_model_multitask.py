import torch
from train_multitask import CustomClassifier
import senteval
from transformers import DebertaV2Tokenizer
import logging
import numpy as np
import argparse
import pandas as pd
import os
import functions_code

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

class SentenceEncoder:
    def __init__(self, device):
        self.device = device
        self.size_embedding = None
        self.pooling_strategy = None
        self.print_best_layers = None
        self.stopwords_set = None

        self.general_embeddings = {}
        self.list_poolings = None
        self.list_layers = None
        self.actual_layer = None

        self.tokenizer = None
        self.model_loaded = None

    def generate_embeddings_from_texts(self, text_list: list, current_task):              
        
        encoding = self.tokenizer.batch_encode_plus(
            text_list,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding='longest',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        embeddings = self.model_loaded.get_embedding(input_ids, attention_mask)
        
        print("Embeddings gerados com sucesso!")
        return embeddings.cpu().numpy()
    
    def _encode(self, sentences, current_task): 
        final_embeddings = self.generate_embeddings_from_texts(sentences, current_task)
        return final_embeddings           
        
def parse_dict_with_eval(value):
    try:
        if isinstance(value, str):
            value = value.replace('np.float64', 'float')
            return eval(value)
        return {}
    except Exception as e:
        return {}

def parse_dict_with_eval_other(value):
    try:
        if isinstance(value, str):
            value = value.replace('np.float64', 'float')
            value = ','.join(value.split(',')[:3]) + '}'
            return eval(value)
        return {}
    except Exception as e:
        return {}
    
def generate_tables_cl(final_df, path_atual, cl_file_name, tasks_cl):
    columns_tasks_cl = tasks_cl
    main_colunas = ['pooling']
    ordem_colunas_cl = main_colunas + columns_tasks_cl        

    path_cl_acc = path_atual + "/" + "cl_acc"
    path_cl_devacc = path_atual + "/" + "cl_devacc"
    os.makedirs(path_cl_acc, exist_ok=True)
    os.makedirs(path_cl_devacc, exist_ok=True)

    data = pd.read_csv(final_df, encoding="utf-8", on_bad_lines="skip")
  
    devacc_data = {'pooling': data['pooling']}
    acc_data = {'pooling': data['pooling']}

    for task in columns_tasks_cl:
        devacc_data[task] = data[task].apply(lambda x: parse_dict_with_eval(x).get('devacc', None))
        acc_data[task] = data[task].apply(lambda x: parse_dict_with_eval(x).get('acc', None))

    devacc_table = pd.DataFrame(devacc_data)
    acc_table = pd.DataFrame(acc_data)   
         

    devacc_table = devacc_table[ordem_colunas_cl]
    acc_table = acc_table[ordem_colunas_cl]  
       

    devacc_table['avg_tasks'] = devacc_table[columns_tasks_cl].mean(axis=1)
    acc_table['avg_tasks'] = acc_table[columns_tasks_cl].mean(axis=1)
       

    devacc_table.to_csv(os.path.join(path_cl_devacc, cl_file_name + '_processado_devacc.csv'))
    acc_table.to_csv(os.path.join(path_cl_acc, cl_file_name + '_processado_acc.csv'))

def save_sent_eval_results(params, filename='sent_eval_embeddings_classification.csv'):
    """
    Salva embeddings, sentenças, tarefas e labels em um CSV.
    Função para fluxo onde o batcher já coleta as labels corretamente.
    """

    # Verificação rápida
    if not (hasattr(params, 'all_embeddings') and hasattr(params, 'all_sentences') and hasattr(params, 'all_labels') and hasattr(params, 'all_tasks')):
        raise ValueError("Faltam embeddings, sentenças, labels ou tasks no params.")

    all_embeddings = np.vstack(params.all_embeddings)
    all_sentences = np.array(params.all_sentences)
    all_tasks = np.array(params.all_tasks)
    all_labels = np.concatenate(params.all_labels)

    # Cria o DataFrame de embeddings
    embedding_dim = all_embeddings.shape[1]
    embedding_columns = [f'dim_{i}' for i in range(embedding_dim)]
    df_embeddings = pd.DataFrame(all_embeddings, columns=embedding_columns)

    # Cria o DataFrame geral
    df = pd.DataFrame({
        'sentence': all_sentences,
        'task': all_tasks,
        'label': all_labels
    })

    # Concatena embeddings ao final
    df = pd.concat([df, df_embeddings], axis=1)

    # Salva CSV
    df.to_csv(filename, index=False)
    print(f"[INFO] CSV salvo com sucesso: {filename}")

def run_senteval(model_name, tasks, epochs, nhid_number, initial_layer_args, final_layer_args, poolings_args, agg_layers_args, type_task, batch_args, optim_args, kfold_args, save_embeddings):
    results_general = {}
    results_general['multitask'] = {}
    results_general['multitask']['out_vec_size'] = 768
    results_general['multitask']['qtd_layers'] = 12
    results_general['multitask']['best_layers'] = 'MULTI'

    device = functions_code.get_device()
    print(f"\nExecuting Device: {device}")

    for t in tasks:
        encoder = SentenceEncoder(device)

        HELD_OUT_TASK = t
        all_task_classes = {"MR": 2, "CR": 2, "MPQA": 2, "SUBJ": 2, "SST2": 2, "TREC": 6, "MRPC": 2}
        training_task_config = {name: num_classes for name, num_classes in all_task_classes.items() if name != HELD_OUT_TASK}

        MODEL_SAVE_PATH = f"/home/yansoares/mlp_finetuning_embedding_models/review/model_exclude_{HELD_OUT_TASK}.bin"

        print("Carregando o modelo e o tokenizador...")
        
        MODEL_NAME = "microsoft/deberta-v3-base"
        
        encoder.model_loaded = CustomClassifier(model_name=MODEL_NAME, task_config=training_task_config).to(device)
        encoder.model_loaded .load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        encoder.tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)
        print(f"Pesos do modelo carregados de {MODEL_SAVE_PATH}")

        senteval_params = {
                'task_path': 'data',
                'usepytorch': True,
                'kfold': kfold_args,
                'classifier': {
                    'nhid': nhid_number,
                    'optim': optim_args,
                    'batch_size': batch_args,
                    'tenacity': 5,
                    'epoch_size': epochs
                },
                'encoder': encoder
            }

        se = senteval.engine.SE(senteval_params, functions_code.batcher, functions_code.prepare)
        results_general['multitask'][HELD_OUT_TASK] = se.eval(HELD_OUT_TASK)
                
    return results_general

def tasks_run(models_args, epochs_args, nhid_args, main_path, initial_layer_args, final_layer_args, poolings_args, agg_layers_args, filename_task, tasks_list, type_task, batch_args, optim_args, kfold_args, save_embeddings):
    path_created = main_path + '/' + filename_task
    os.makedirs(path_created, exist_ok=True)

    logging.basicConfig(
        filename=path_created + '/' + filename_task + '_log.txt',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    results_data = []

    for model_name in models_args:
        print(f"\nExecuting Model: {model_name}")
        results = run_senteval(model_name, tasks_list, epochs_args, nhid_args, initial_layer_args, final_layer_args, poolings_args, agg_layers_args, type_task, batch_args, optim_args, kfold_args, save_embeddings)
        for pooling, res in results.items():
            if type_task == 'cl':
                dict_results = [res.get(task, {}) for task in tasks_list]
            elif type_task == 'si':
                dict_results = [res.get(task, {}).get('all', 0) for task in tasks_list[:5]] + [res.get(task, {}) for task in tasks_list[-2:]]
            
            results_data.append({
                "model": model_name,
                "pooling": pooling,
                "out_vec_size": res.get('out_vec_size'),
                "best_layers": res.get('best_layers'),
                "epochs": epochs_args,
                "nhid": nhid_args,
                "qtd_layers": res.get('qtd_layers'),              
                **{task: dict_results[i] for i, task in enumerate(tasks_list)}
            })
        
        final_df1 = pd.DataFrame(results_data)
        final_df1.to_csv(path_created + '/' + filename_task + '_intermediate.csv', index=False)
                    
    final_df = pd.DataFrame(results_data)
    final_df.to_csv(path_created + '/' + filename_task + '.csv', index=False)
    generate_tables_cl(path_created + '/' + filename_task + '.csv', path_created, filename_task, tasks_list)

def main():
    parser = argparse.ArgumentParser(description="SentEval Experiments")
    parser.add_argument("--task_type", type=str, default='classification', choices=['classification', 'similarity'], help="Tipo de tarefa (classification ou similarity)")
    parser.add_argument("--models", type=str, default='deberta-base', help="Modelos separados por vírgula (sem espaços)")
    parser.add_argument("--epochs", type=int, default=1, help="Número máximo de épocas do classificador linear")
    parser.add_argument("--batch", type=int, default=1024, help="Batch Size do classificador")
    parser.add_argument("--kfold", type=int, default=10, help="KFold para validação")
    parser.add_argument("--optim", type=str, default='adam', help="otimizador do classificador")
    parser.add_argument("--nhid", type=int, default=0, help="Numero de camadas ocultas (0 = Logistic Regression, 1 ou mais = MLP)")
    parser.add_argument("--initial_layer", type=int, default=0, help="Camada inicial para execução dos experimentos (default metade superior)")
    parser.add_argument("--final_layer", type=int, default=12, help="Camada inicial para execução dos experimentos (default metade superior)")
    parser.add_argument("--poolings", type=str, default="multitask", help="Poolings separados por virgula (sem espacos) ou simple, simple-ns, two, three")
    parser.add_argument("--agg_layers", type=str, default="teste_antes_terminar", help="agg layers separados por virgula (sem espacos)")
    parser.add_argument("--tasks", type=str, help="tasks separados por virgula (sem espacos)")
    parser.add_argument("--save_embeddings", type=int, default=0, help="1 - yes, 0 - no")
    args = parser.parse_args()

    task_type_args = args.task_type 
    models_args = args.models.split(",")        
    epochs_args = args.epochs 
    batch_args = args.batch 
    kfold_args = args.kfold     
    optim_args = args.optim 
    nhid_args = args.nhid
    initial_layer_args = args.initial_layer 
    final_layer_args = args.final_layer
    poolings_args = args.poolings.split(",")
    agg_layers_args = args.agg_layers.split(",")  
    save_embeddings = int(args.save_embeddings)

    main_path = 'results_tables'   

    initial_layer_args_print = args.initial_layer if args.initial_layer is not None else "default"
    final_layer_args_print = args.final_layer if args.final_layer is not None else "default"

    '''filename_task = ('_models_' + '&'.join([st for st in models_args]) + 
                     '_epochs_' + str(epochs_args) + 
                     '_batch_' + str(batch_args) +
                     '_kfold_' + str(kfold_args) +
                     '_optim_' + str(optim_args) +
                     '_nhid_' + str(nhid_args) + 
                     '_initiallayer_' + str(initial_layer_args_print) + 
                     '_finallayer_' + str(final_layer_args_print) +
                     '_pooling_' + '&'.join([st for st in poolings_args]) + 
                     '_agglayers_' + '&'.join([st for st in agg_layers_args])
                     )'''
    filename_task = ('_models_' + '&'.join([st for st in models_args]) + 
                     '_epochs_' + str(epochs_args) + 
                     '_batch_' + str(batch_args) +
                     '_kfold_' + str(kfold_args) +
                     '_optim_' + str(optim_args) +
                     '_nhid_' + str(nhid_args) + 
                     '_pooling_' + '&'.join([st for st in poolings_args]) + 
                     '_agglayers_' + '&'.join([st for st in agg_layers_args])
                     )

    if task_type_args == "classification":      
        filename_cl = "cl" + filename_task
        #classification_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']        
        classification_tasks = ['MR', 'CR']
        classification_tasks = args.tasks.split(",") if args.tasks is not None else classification_tasks
        tasks_run(models_args, epochs_args, nhid_args, main_path, initial_layer_args, final_layer_args, poolings_args, agg_layers_args, filename_cl, classification_tasks, 'cl', batch_args, optim_args, kfold_args, save_embeddings)

    elif task_type_args == "similarity":
        filename_si = "si" + filename_task
        similarity_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        similarity_tasks = args.tasks.split(",") if args.tasks is not None else similarity_tasks
        tasks_run(models_args, epochs_args, nhid_args, main_path, initial_layer_args, final_layer_args, poolings_args, agg_layers_args, filename_si, similarity_tasks, 'si', batch_args, optim_args, kfold_args, save_embeddings)

if __name__ == "__main__":
    main()