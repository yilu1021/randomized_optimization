from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import mlrose_hiive as mlrose
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
class NN:
    def __init__(self):
        digits = datasets.load_digits()
        n_samples = len(digits.images)
        self.data = digits.images.reshape((n_samples, -1))
        self.targets = digits.target

    def get_ga_model(self):
        return {'model':mlrose.NeuralNetwork(hidden_nodes = [90], activation = 'relu', algorithm = 'genetic_alg',
                                         max_iters = 2000, bias = True, is_classifier = True, learning_rate = 1,
                                         early_stopping = True, clip_max = 1, max_attempts = 2000, random_state = 3,
                                        curve=True, pop_size=50), 'name': 'GA', 'nn': False}

    def get_rhc_model(self):
        return {'model':mlrose.NeuralNetwork(hidden_nodes = [90], activation = 'relu', algorithm = 'random_hill_climb',
                                         max_iters = 2000, bias = True, is_classifier = True, learning_rate = 1,
                                         early_stopping = True, clip_max = 1, max_attempts = 2000, random_state = 3,
                                        curve=True), 'name': 'RHC', 'nn': False}
    def get_sa_model(self, learning_rate):
        return {'model':mlrose.NeuralNetwork(hidden_nodes = [90], activation = 'relu', algorithm = 'simulated_annealing', max_iters = 2000, bias = True, is_classifier = True, learning_rate = learning_rate,
                                         early_stopping = True, clip_max = 1, max_attempts = 2000, random_state = 3,
                                        curve=True), 'name': 'SA', 'nn': False}

    def get_gradient_descent_model(self):
        return {'model': mlrose.NeuralNetwork(hidden_nodes = [90], activation = 'relu', algorithm = 'gradient_descent',
                                         max_iters = 2000, bias = True, is_classifier = True, learning_rate = 0.0001,
                                         early_stopping = True, clip_max = 1, max_attempts = 2000, random_state = 3,
                                            curve=True), 'name': 'Gradient Descent', 'nn': True}

    def fit_predict(self, nn_model):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.targets, test_size=0.2, random_state=3,
                                                            stratify=self.targets)
        one_hot = OneHotEncoder()
        y_train = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
        y_test = one_hot.transform(y_test.reshape(-1, 1)).todense()
        nn_model.fit(X_train, y_train)
        y_train_pred = nn_model.predict(X_train)
        y_train_accuracy = accuracy_score(y_train, y_train_pred)

        y_test_pred = nn_model.predict(X_test)
        y_test_accuracy = accuracy_score(y_test, y_test_pred)

        return y_train_accuracy, y_test_accuracy

    def plot(self, models):
        title1 = 'loss_over_iteration'
        plt.title(title1)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        for model in models:
            fitness = model['model'].fitness_curve
            if model['nn'] is True:
                fitness = np.array(fitness) * -1
            plt.plot(range(0, len(model['model'].fitness_curve)), fitness, label=model['name'])

        plt.legend(loc='best')
        plt.savefig('output_part2/' + title1 + '.png')
        plt.close()

        title2='training_accuracy'
        x_pos = [i for i, _ in enumerate(models)]
        y_train_accuracies = []
        names = []
        for model in models:
            y_train_accuracies.append(model['y_train_accuracy'])
            names.append(model['name'])
        plt.bar(x_pos, y_train_accuracies, color='green')
        plt.xlabel("Classifier Type")
        plt.ylabel("Training Accuracy")
        plt.ylim(0, 1.1)
        plt.title("Training Accuracy Comparison")
        plt.xticks(x_pos, names)

        for i in range(len(y_train_accuracies)):
            plt.annotate(str("{:.2%}".format(y_train_accuracies[i])), xy=(x_pos[i], y_train_accuracies[i]))

        plt.legend(loc='best')
        plt.savefig('output_part2/' + title2 + '.png')
        plt.close()

        title3 = 'test_accuracy'
        x_pos = [i for i, _ in enumerate(models)]
        y_test_accuracies = []
        names = []
        for model in models:
            y_test_accuracies.append(model['y_test_accuracy'])
            names.append(model['name'])
        plt.bar(x_pos, y_test_accuracies, color='green')
        plt.xlabel("Classifier Type")
        plt.ylabel("Test Accuracy")
        plt.ylim(0, 1.1)
        plt.title("Test Accuracy Comparison")
        plt.xticks(x_pos, names)

        for i in range(len(y_test_accuracies)):
            plt.annotate(str("{:.2%}".format(y_test_accuracies[i])), xy=(x_pos[i], y_test_accuracies[i]))

        plt.legend(loc='best')
        plt.savefig('output_part2/' + title3 + '.png')
        plt.close()


    def run(self):
        # ga = self.get_ga_model()
        # rhc = self.get_rhc_model()
        # sa = self.get_sa_model(learning_rate=1)
        # gradient_descent = self.get_gradient_descent_model()
        #
        # models = [ga, rhc, sa, gradient_descent]
        # for model in models:
        #     y_train_accuracy, y_test_accuracy = self.fit_predict(model['model'])
        #     model['y_train_accuracy'] = y_train_accuracy
        #     model['y_test_accuracy'] = y_test_accuracy
        #
        # self.plot(models)


        sa_learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

        title1 = 'sa_learning_rate_loss_over_iteration'
        plt.title(title1)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        for learning_rate in sa_learning_rates:
            model = self.get_sa_model(learning_rate)
            y_train_accuracy, y_test_accuracy = self.fit_predict(model['model'])
            fitness = model['model'].fitness_curve
            plt.plot(range(0, len(model['model'].fitness_curve)), fitness, label=str(learning_rate))

        plt.legend(loc='best')
        plt.savefig('output_part2/' + title1 + '.png')
        plt.close()





