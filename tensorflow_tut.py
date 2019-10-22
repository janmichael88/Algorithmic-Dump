import tensorflow as tf
import os as os
from skimage import transform
from skimage.color import rgb2gray


#initilize two constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

#multiply them 
result = tf.multiply(x1,x2)

#initilize the session
sess = tf.Session()

#print the result
print(sess.run(result))

#close the session 
sess.close()

#run separately, initialize session and run result
with tf.Session() as sess:
	output = sess.run(result)
	print(output)

#use config protocol 
config = tf.ConfigProto(log_device_replacement=True)
config=tf.ConfigProto(allow_soft_placement=True)

#beginning of the tutorial
def load_data(data_directory):
	#set up list of directories
	directories = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory,d))]
	#initliaze labels and images
	labels = []
	images = []

	#loop of each file in directory
	for d in directories:
		#assignments
		label_directory = os.path.join(data_directory,d)
		file_names = [os.path.join(label_directory,f) for f in os.listdir(label_directory) if f.endswith(".ppm")]
		for f in file_names:
			images.append(skimage.data.imread(f))
			labels.append(int(d))
	return(images,labels)

#set root path
path = "/Users/janmichaelaustria/Documents/Python/Random Projects/tf_stopsign_tutorial"
#get directories for training and testing
train_data_directory = os.path.join(path, "Training")
test_data_directory = os.path.join(path, "Testing")

#get images and labels from train
images_train, labels_train = load_data(train_data_directory)
#images were converted to greyscale, each of dim 3
#plot the labels
import matplotlib.pyplot as plt 
plt.hist(labels_train,62)
plt.show()

#xexamine randome images
traffic_signs = [300, 2250, 3650,4000]
for i in range(len(traffic_signs)):
	plt.subplot(1,4,i+1)
	plt.axis('off')
	plt.imshow(images_train[traffic_signs[i]])
	plt.subplots_adjust(wspace=0.5)

plt.show()

#images are not all the sam size
#plot again but and notice dimension sizes

#show sample image for each class
unique_labels = set(labels_train)
#initialize the figure
plt.figure(figsize=(17,17))
#set counter
count = 1
#loop over unique lables
for label in unique_labels:
	#get image
	image = images_train[labels_train.index(label)]
	#define subplots
	plt.subplot(8,8,count)
	plt.axis('off')
	#add title
	plt.title("Label {0} ({1})".format(label,labels_train.count(label)))
	#increment counter
	count += 1
	plt.imshow(image)

#need to rescale images and convert to greyscale
#reshape to 28 and conver to gresysclae
from skimage import transform
from skimage.color import rgb2gray
images28 = rgb2gray(np.array([transform.resize(image,(28,28)) for image in images_train]))


#note the min and max values have also changes
#check that images are actually gray
traffic_signs = [300, 2250, 3650, 4000]

for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images28[traffic_signs[i]], cmap="gray")
    plt.subplots_adjust(wspace=0.5)

#convert into a function

def show_random_images(n,arr_images):
	#show random images from an array of images
	max_n = np.max(n)
	min_n = np.min(n)
	#generate random numbers of length n up to max_N
	img_to_show = np.random.randint(0,len(arr_images),n)

	for i in range(0,n):
		plt.subplot(1, n, i+1)
		plt.axis('off')
		plt.imshow(arr_images[img_to_show[i]], cmap="gray")
		plt.subplots_adjust(wspace=0.5)
	plt.show()

#begin nueral network for image classification
#set input shape
#initlizae the input data
sparse_softmax_cross_entropy_with_logits()
# Initialize placeholders 
x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28])
y = tf.placeholder(dtype = tf.int32, shape = [None])
#flatten the input data
# Flatten the input data
images_flat = tf.contrib.layers.flatten(x)
# Fully connected layer 
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)
# Define a loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, 
                                                                    logits = logits))
# Define an optimizer 
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
# Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)
# Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", correct_pred)

#being training the nueral network
tf.set_random_seed(1234)
sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(201):
        print('EPOCH', i)
        _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images28, y: labels_train})
        if i % 10 == 0:
            print("Loss: ", loss)
        print('DONE WITH EPOCH')

#evaluate network
# Import `matplotlib`
import matplotlib.pyplot as plt
import random

# Pick 10 random images
sample_indexes = random.sample(range(len(images28)), 10)
sample_images = [images28[i] for i in sample_indexes]
sample_labels = [labels_train[i] for i in sample_indexes]

# Run the "correct_pred" operation
predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]
                        
# Print the real and predicted labels
print(sample_labels)
print(predicted)

# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction), 
             fontsize=12, color=color)
    plt.imshow(sample_images[i],  cmap="gray")

plt.show()


#compare with test data
# Import `skimage`
from skimage import transform
# Load the test data
test_images, test_labels = load_data(test_data_directory)
# Transform the images to 28 by 28 pixels
test_images28 = [transform.resize(image, (28, 28)) for image in test_images]
# Convert to grayscale
from skimage.color import rgb2gray
test_images28 = rgb2gray(np.array(test_images28))
# Run predictions against the full test set.
predicted = sess.run([correct_pred], feed_dict={x: test_images28})[0]
# Calculate correct matches 
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])
# Calculate the accuracy
accuracy = match_count / len(test_labels)
# Print the accuracy
print("Accuracy: {:.3f}".format(accuracy))

#remember to close session
sess.clost()



















































