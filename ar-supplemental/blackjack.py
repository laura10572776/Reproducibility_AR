import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from stoch_mcts.game.game import ContinuousBlackJack
from stoch_mcts.mcts.mcts import MCTS
from stoch_mcts.mcts.selector import ProgressiveWidening, AbstractionRefining, sampler
from stoch_mcts.mcts.backpropagator import *
from stoch_mcts.trainer.trainer import run
from stoch_mcts.agent.agent import Agent
from stoch_mcts.mcts.selector import *
from stoch_mcts.mcts.evaluator import *

# Parameters
n_trials = 1000
ground_truth_n_samples = 100_000
num_rollouts = [10, 20, 50, 100, 200, 500, 1000]
k_values = [1, 2, 3, 4]
pw_exponents = [0.2, 0.5, 0.8]
ar_exponent = 0.1
ar_factor = 2

game = ContinuousBlackJack()
mcts = MCTS(
        selector=Selector(action_selector=dealer_policy,
                          state_selector=sampler),
        evaluator=null,
        backpropagator=TreeBackpropagator(discount=1)
    )

print("Compute ground truth")
result = run(game, Agent(mcts), n_trials, ground_truth_n_samples)
print(len(result))
ground_truth = np.mean(result)
print(ground_truth)

# Initialize the DataFrame
df = {'method': [], 'value_error': [], 'num_rollouts': [], 'k': [], 'exp': [], 'time_minutes': []}
df_ = pd.DataFrame(df)
df_['value_error'] = df_['value_error'].astype(float)

for trial in tqdm(range(5)):
    game = ContinuousBlackJack()  # Initialize the game
    for nr in num_rollouts:
        for k in k_values:
            for alpha in pw_exponents:
                mcts = MCTS(
                    selector=Selector(action_selector=dealer_policy, state_selector=ProgressiveWidening(k=k, alpha=alpha)),
                    evaluator=null,
                    backpropagator=TreeBackpropagator(discount=1)
                )
                # Measure the start time
                start_time = time.time()

                # Progressive Widening simulation
                result = run(game, Agent(mcts), n_trials, nr)

                # Measure the end time
                end_time = time.time()
                # Calculate the time taken in minutes
                time_taken = (end_time - start_time) / 60

                error = np.mean(result) - ground_truth
                df['method'].append(f'PW{alpha}')
                df['exp'].append(alpha)
                df['k'].append(k)
                df['num_rollouts'].append(nr)
                df['value_error'].append(np.abs(error))
                df['time_minutes'].append(time_taken)
                df_ = pd.DataFrame(df)
                df_['value_error'] = df_['value_error'].astype(float)
                # Carica un DataFrame da un file CSV e concatenalo con 'df_'
                try:
                    df_csv = pd.read_csv(f"df/df_{k_values[0]}_bj_prova.csv")
                    df_csv.drop_duplicates(inplace=True)
                    df_ = pd.concat([df_, df_csv], ignore_index=True)
                except Exception as e:
                    print("Errore nella lettura o concatenazione del CSV:", e)

                # Salva il DataFrame risultante
                try:
                    df_.to_csv(f"df/df_{k_values[0]}_bj.csv", index=False)
                except Exception as e:
                    print("Errore nel salvataggio del CSV:", e)

        mcts = MCTS(
            selector=Selector(action_selector=dealer_policy,
                              state_selector=AbstractionRefining(k=ar_exponent, alpha=ar_factor, dist=game.distance)),
            evaluator=null,
            backpropagator=TreeBackpropagator(discount=1)
        )

        game = ContinuousBlackJack()
        print("AbstractionRefining")
        # Measure the start time
        start_time = time.time()
        result = run(game, Agent(mcts), n_trials, nr)

        end_time = time.time()
        # Calculate the time taken in minutes
        time_taken = (end_time - start_time) / 60
        error = np.mean(result) - ground_truth
        df['method'].append('AR')
        df['k'].append(ar_factor)
        df['exp'].append(ar_exponent)
        df['num_rollouts'].append(nr)
        df['value_error'].append(np.abs(error))
        df['time_minutes'].append(time_taken)
        df_ = pd.DataFrame(df)
        df_['value_error'] = df_['value_error'].astype(float)
        df_.to_csv(f"df/df_{k_values[0]}_bj.csv", index=False)

df_ = pd.DataFrame(df)
df_['value_error'] = df_['value_error'].astype(float)
df_ = pd.concat([df_, pd.read_csv(f"df/df_{k_values[0]}_bj_prova.csv")])
df_.drop_duplicates(inplace=True)
df_.to_csv(f"df/df_{k_values[0]}_bj.csv", index=False)
df_ = pd.DataFrame()

new_rows = pd.DataFrame()
ar_rows = df_[df_['method'] == 'AR']

# Duplicate rows with 'AR' method and change 'k' values
for k in k_values:
    temp = ar_rows.copy()
    temp['k'] = k
    new_rows = pd.concat([new_rows, temp])

# Add the new rows to the original DataFrame
df_ = pd.concat([df_, new_rows], ignore_index=True)

# Delete rows where 'method' is 'AR' and 'k' is equal to 'ar_factor'
df_ = df_.drop(df_[(df_['method'] == 'AR') & (df_['k'] == ar_factor)].index)

sns.relplot(data=df_, x='num_rollouts', y='value_error', hue='method', kind='line', col='k',
            facet_kws={'sharey': True, 'sharex': True}).set(yscale="log")
plt.savefig('blackjack_value_error_analysis.png')
plt.show()
