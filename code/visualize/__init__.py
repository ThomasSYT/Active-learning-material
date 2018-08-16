"""Visualization handlers
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot(iteration, acc):
    """Simple function for plotting two lists of result and scores.
    """
    plt.plot(iteration,acc)
    plt.ylabel('Accuracy')
    plt.xlabel('Iteration')
    plt.show()

def plot_several(results, captions, upper_bound):
    """Function for plotting several results. 
    Input is a list of lists of (iteration, score) pairs.
    """
    plt.figure(1)
    handles = [] # Handles for the legend
    # Add upper bound to the plot
    plt.axhline(y=upper_bound, color='grey', linestyle='-')
    handles.append(mpatches.Patch(color='grey', label="Upper bound"))
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black'] # Our colors for the plots 
    for i, (single_result, caption) in enumerate(zip(results, captions)):
        # Add the results for a single active learning approach and set the legend
        iteration, score = [pair[0] for pair in single_result], [pair[1] for pair in single_result]
        plt.plot(iteration,score, color=colors[i])
        handles.append(mpatches.Patch(color=colors[i], label=caption))
    plt.legend(handles=handles)
    plt.show()



