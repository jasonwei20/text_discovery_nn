from nlp_utils import *
import math

val_acc_e = [0.6675, 0.65, 0.7325, 0.708]
entropy_e = [0.0478, 0.0147, 0.00428, 0.00999]
val_acc_b = [0.7725, 0.718, 0.783, 0.798]
entropy_b = [0.0622, 0.0437, 0.0622, 0.0597]

labels = ['nb', 'rf', 'lr', 'nn']

fig, ax = plt.subplots()

if __name__ == "__main__":

	group_e = 'embeddings'
	#embeddings
	for i in range(len(val_acc_e)):
		x = math.log(1/entropy_e[i])
		y = val_acc_e[i]
		ax.scatter(x, y, color='b')

	group_b = 'bag of words'
	#bag of words
	for i in range(len(val_acc_b)):
		x = math.log(1/entropy_b[i])
		y = val_acc_b[i]
		ax.scatter(x, y, color='g')

	for i, txt in enumerate(labels):
		ax.annotate(txt, (math.log(1/entropy_e[i]), val_acc_e[i]), fontsize = 10)
		ax.annotate(txt, (math.log(1/entropy_b[i]), val_acc_b[i]), fontsize = 10)
	
	ax.set_title("Performance and Entropy of Various Classifiers and Representations", fontsize=12)

	plt.xlabel(r'Entropy $(log(\frac{1}{\gamma}))$')
	plt.ylabel('Validation Accuracy')
	plt.savefig("outputs/ablation_test.png", dpi=400)