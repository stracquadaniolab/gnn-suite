#!/usr/bin/env python3
import sys
import torch
import optuna
sys.path.append('models.py')
import typer

from gnn import run

import os
import sys
import contextlib

def run_silently(func, *args, **kwargs):
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            return func(*args, **kwargs)

def objective_gnn(trial, model_name, gene_filename, network_filename, num_epochs=300):
    # Define hyperparameters to optimize

    dropout = trial.suggest_float('dropout', 0.0, 0.8)
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 5e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 5e-1, log=True)
    
    # Run the model with the hyperparameters
    max_bacc = run(
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

    return max_bacc

def objective_gcn2(trial, model_name, gene_filename, network_filename, num_epochs=300):
    # Define hyperparameters to optimize
    alpha = trial.suggest_float('alpha', 1e-3, 10.0, log=True)
    theta = trial.suggest_float('theta', 1e-3, 10.0, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.8)
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 5e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 5e-1, log=True)
    
    # Run the model with the hyperparameters
    max_bacc = run(
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
    
    return max_bacc


def run_optuna(data_pair, model):
    gene_filename = data_pair['geneFile']
    network_filename = data_pair['networkFile']
    data_name = data_pair['name']
    model_name = model

    # Set the number of epochs for training
    num_epochs = 250
    
    # Create a study object
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


def run_hyperopt(
        gene_filename: str,
        network_filename: str,
        model_name: str,
        data_set: str):

    data_pairs = [{'name': data_set, 
                   'networkFile': network_filename, 
                   'geneFile': gene_filename}]

    models = [model_name]
    
    for data_pair in data_pairs:
        for model in models:
            print(f"Running hyperparameter optimization for model '{model}' with data pair '{data_pair['name']}'")
            run_optuna(data_pair, model)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    typer.run(run_hyperopt)
    

