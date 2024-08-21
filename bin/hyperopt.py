# This script is used to run hyperparameter optimization for all models on all datasets
import sys
import torch
import optuna
sys.path.append('models.py')

from gnn import run

def objective_gnn(trial, model_name, gene_filename, network_filename, num_epochs=300):
    # Define hyperparameters to optimize

    dropout = trial.suggest_float('dropout', 0.0, 0.8)
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 5e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 5e-1, log=True)
    
    # Run the model with the hyperparameters
    bacc = run(
        gene_filename = gene_filename,
        network_filename = network_filename,
        train_size = 0.8,
        model_name= model_name,
        epochs= num_epochs,
        learning_rate = learning_rate,
        weight_decay= weight_decay,
        eval_threshold= 0.9,
        verbose_interval= 10,
        dropout= dropout
    )
    
    return bacc

def objective_gcn2(trial, model_name, gene_filename, network_filename, num_epochs=300):
    # Define hyperparameters to optimize
    alpha = trial.suggest_float('alpha', 1e-3, 10.0, log=True)
    theta = trial.suggest_float('theta', 1e-3, 10.0, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.8)
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 5e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 5e-1, log=True)
    
    # Run the model with the hyperparameters
    bacc = run(
        gene_filename = gene_filename,
        network_filename = network_filename,
        train_size = 0.8,
        model_name= model_name,
        epochs= num_epochs,
        learning_rate = learning_rate,
        weight_decay= weight_decay,
        eval_threshold= 0.9,
        verbose_interval= 10,
        dropout= dropout,
        alpha= alpha,
        theta= theta
    )
    
    return bacc


def run_optuna(data_pair, model):
    gene_filename = data_pair['geneFile']
    network_filename = data_pair['networkFile']
    data_name = data_pair['name']
    model_name = model

    num_epochs = 250
    
    study = optuna.create_study(study_name=model_name+"_hp_search",
                                direction="maximize")
    

    if model_name == "gcn2":
        study.optimize(lambda trial: objective_gcn2(trial,
                                                    model_name,
                                                    gene_filename,
                                                    network_filename,
                                                    num_epochs),
                                                n_jobs=-1,
                                                n_trials=300)
    else:
        study.optimize(lambda trial: objective_gnn(trial, model_name,
                                                   gene_filename,
                                                    network_filename,
                                                    num_epochs),
                                                n_jobs=-1,
                                                n_trials=300)


    # Print the best trial
    best_trial = study.best_trial
    print("Best trial:")
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Save the best trial to a file
    save_dir = '/home/essharom/code/cancer-gnn-nf/results/hyperparameters'

    # CHANGE FILENAME TO DIFFERENT MODELS
    # Save the best trial to a file
    with open(f'{save_dir}/best_trial_{model_name}_{data_name}.txt', 'w') as f:
        f.write("Best trial:\n")
        f.write(f"  Value: {best_trial.value}\n")
        f.write("  Params:\n")
        for key, value in best_trial.params.items():
            f.write(f"    {key}: {value}\n")


# Define the directory to save hyperparameter results
save_dir = '/home/essharom/code/cancer-gnn-nf/results/hyperparameters'
output_csv_file = '/home/essharom/code/cancer-gnn-nf/results/hyperparameters.csv'

def save_best_trial(study, model_name, data_name, save_dir=save_dir):
    best_trial = study.best_trial

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save the best trial to a file with the appropriate name
    filename = f'best_trial_{model_name}_{data_name}.txt'
    file_path = os.path.join(save_dir, filename)
    
    with open(file_path, 'w') as f:
        f.write("Best trial:\n")
        f.write(f"  Value: {best_trial.value}\n")
        f.write("  Params:\n")
        for key, value in best_trial.params.items():
            f.write(f"    {key}: {value}\n")
    
    print(f"Best trial saved to {file_path}")

def extract_model_dataset_names(filename):
    parts = filename.split("_")
    model_name = parts[2]
    dataset_name = "_".join(parts[3:]).split(".")[0]
    return model_name, dataset_name

def handle_special_characters(value):
    numeric_value = re.findall(r'[+-]?\d+(?:\.\d+)?', value)
    if numeric_value:
        return float(numeric_value[0])
    else:
        return value

def summarize_results_to_csv(directory_path=save_dir, output_file=output_csv_file):
    # Initialize an empty list to store the data
    data = []

    # Loop through the files in the directory
    for filename in os.listdir(directory_path):
        if filename.startswith("best_trial"):
            model_name, dataset_name = extract_model_dataset_names(filename)
            with open(os.path.join(directory_path, filename), 'r') as file:
                lines = file.readlines()
                # Extract the value using regular expressions to handle special characters
                bacc_value = handle_special_characters(re.findall(r'[+-]?\d+(?:\.\d+)?', lines[1])[0])
                params = {}
                for line in lines[3:]:
                    key, value = line.split(":")
                    params[key.strip()] = handle_special_characters(value.strip())
                data.append({
                    'Model': model_name,
                    'Dataset': dataset_name,
                    'BACC': bacc_value,
                    **params
                })

    # Create a DataFrame from the data list
    df = pd.DataFrame(data)

    # Save the data table to a CSV file
    df.to_csv(output_file, index=False)

    print("Data table has been saved to:", output_file)

def main():
    # Example setup - Replace with actual data and model handling logic
    data_pairs = [{'name': 'example_dataset'}]  # This should be your actual data
    models = ['example_model']  # Replace with your model names
    
    for data_pair in data_pairs:
        data_name = data_pair['name']  # Get dataset name
        for model_name in models:
            print(f"Running hyperparameter optimization for model '{model_name}' with data pair '{data_name}'")
            
            # Assuming run_optuna is your function to run the optimization
            study = run_optuna(data_pair, model_name)  # Replace with actual function call
            save_best_trial(study, model_name, data_name)
            
            # Clear CUDA cache after each run to avoid memory issues
            torch.cuda.empty_cache()

    # After running all optimizations, summarize the results into a CSV
    summarize_results_to_csv()

if __name__ == "__main__":
    main()
