import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from stoch_mcts.game.game import Trap
from stoch_mcts.mcts.mcts import MCTS
from stoch_mcts.mcts.selector import ProgressiveWidening, AbstractionRefining, sampler
from stoch_mcts.mcts.backpropagator import *
from stoch_mcts.trainer.trainer import run
from stoch_mcts.agent.agent import Agent
from stoch_mcts.mcts.selector import *
from stoch_mcts.mcts.evaluator import *
import matplotlib.colors as mcolors

# Parameters
n_trials = 1000
num_rollouts = [10, 20, 50, 100, 200, 500, 1000]
k_values = [1]  # , 0.05, 1, 2]
#pw_exponents = [0.7, 0.8, 1]  # k = 0.02
# pw_exponents = [0.5, 0.6, 0.7, 0.8, 0.9] k = 0.01
pw_exponents = [0.15, 0.3, 0.5, 0.7] #k = 1
#pw_exponents = [0.05, 0.15, 0.2, 0.3]
ar_exponent = 0.1
ar_factor = 0.1
# Initialize the DataFrame
df = {'method': [], 'expected_return': [], 'num_rollouts': [], 'k': [], 'exp': [], 'time_minutes': []}

for trial in tqdm(range(10)):
    game = Trap()
    for nr in num_rollouts:
        for k in k_values:
            for alpha in pw_exponents:
                mcts = MCTS(
                    selector=Selector(action_selector=UCT(100), state_selector=ProgressiveWidening(k=k, alpha=alpha)),
                    evaluator=RandomRollouts(1, 1000),
                    backpropagator=TreeBackpropagator(discount=1)
                )
                # Progressive Widening simulation
                print(f"alpha is equal to {alpha} with nr equal to {nr}")
                # Measure the start time
                start_time = time.time()

                # Progressive Widening simulation
                result = run(game, Agent(mcts), n_trials, nr)

                # Measure the end time
                end_time = time.time()

                # Calculate the time taken in minutes
                time_taken = (end_time - start_time) / 60

                df['method'].append(f'PW{alpha}')
                df['exp'].append(alpha)
                df['k'].append(k)
                df['num_rollouts'].append(nr)
                df['expected_return'].append(np.mean(result))
                df['time_minutes'].append(time_taken)

                df_ = pd.DataFrame(df)
                try:
                    df_['expected_return'] = df_['expected_return'].astype(float)
                except Exception as e:
                    print("Errore nella conversione di 'expected_return':", e)

                # Verifica il contenuto di 'k_values'
                print("k_values[0]:", k_values[0])

                # Carica un DataFrame da un file CSV e concatenalo con 'df_'
                try:
                    df_csv = pd.read_csv(f"df/df_{k_values[0]}_n.csv")
                    df_csv.drop_duplicates(inplace=True)
                    df_ = pd.concat([df_, df_csv], ignore_index=True)
                except Exception as e:
                    print("Errore nella lettura o concatenazione del CSV:", e)

                # Salva il DataFrame risultante
                try:
                    df_.to_csv(f"df/df_{k_values[0]}_n.csv", index=False)
                except Exception as e:
                    print("Errore nel salvataggio del CSV:", e)


    mcts = MCTS(
        selector=Selector(action_selector=UCT(100),
                          state_selector=AbstractionRefining(k=ar_exponent, alpha=ar_factor, dist=game.distance)),
        evaluator=RandomRollouts(1, 1000),
        backpropagator=TreeBackpropagator(discount=1)
    )
    game = Trap()

    print("AbstractionRefining")
    error = run(game, Agent(mcts), n_trials, nr)
    df['method'].append('AR')
    df['k'].append(ar_factor)
    df['exp'].append(ar_exponent)
    df['num_rollouts'].append(nr)
    df['expected_return'].append(np.mean(error))


# Convert to DataFrame and visualize the results
df_ = pd.DataFrame(df)
df_['expected_return'] = df_['expected_return'].astype(float)
df_ = pd.concat([df_, pd.read_csv(f"df/df_{k_values[0]}.csv")], ignore_index=True)
df_.to_csv(f"df/df_{k_values[0]}.csv", index = False)
sns.relplot(data=df_, x='num_rollouts', y='expected_return', hue='exp', kind='line', col='k',
            facet_kws={'sharey': True, 'sharex': True})
plt.savefig(f'trap_analysis_{k_values[0]}_n.png')

plt.show()
