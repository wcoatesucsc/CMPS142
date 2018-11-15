package cmps142_hw4;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class LogisticRegression {

        /** the learning rate */
        private double rate=0.01;

        /** the weights to learn */
        private double[] weights;

        /** the number of iterations */
        private int ITERATIONS = 200;

        /** TODO: Constructor initializes the weight vector. Initialize it by setting it to the 0 vector. **/
        public LogisticRegression(int n) { // n is the number of weights to be learned
           weights = new double[n];

           for(int i = 0; i < n; i++){
              weights[i] = 0;
           }
        }

        /** TODO: Implement the function that returns the L2 norm of the weight vector **/
        // L2 norm = square root of sum of squared vector values
        private double weightsL2Norm(){
           double norm = 0;
           for(int i = 0; i < weights.length; i++){
              norm += Math.pow(weights[i], 2);
           }
           return Math.sqrt(norm);
        }

        /** TODO: Implement the sigmoid function **/
        // P(Y = 1|X) = 1/(1 + e^(-w.x))
        // P(Y = 1|X) = 1/(1 + e^(-z))
        private static double sigmoid(double z) {
           // Java has built in Math.exp(double a) function: returns e^a
           double denominator = 1 + Math.exp(-z); 
           return (1/denominator);
        }

        /** TODO: Helper function for prediction **/
        /** Takes a test instance as input and outputs the probability of the label being 1 **/
        /** This function should call sigmoid() **/
        // Take the dot product of the weight vector and the instance vector (x)
        // and call sigmoid() on that double
        private double probPred1(double[] x) {
           double dotProduct = 0;
           for(int i = 0; i < weights.length; i++){
              dotProduct += (weights[i] * x[i]);
           }
           return sigmoid(dotProduct);
        }

        /** TODO: The prediction function **/
        /** Takes a test instance as input and outputs the predicted label **/
        /** This function should call probPred1() **/
        public int predict(double[] x) {
           if(probPred1(x) >= 0.5){
              return 1;
           }
           else{
              return 0;
           }
           
        }

        /** This function takes a test set as input, call the predict() to predict a label for it, and prints the accuracy, P, R, and F1 score of the positive class and negative class and the confusion matrix **/
        public void printPerformance(List<LRInstance> testInstances) {
            double acc = 0;
            double p_pos = 0, r_pos = 0, f_pos = 0;
            double p_neg = 0, r_neg = 0, f_neg = 0;
            int TP=0, TN=0, FP=0, FN=0; // TP = True Positives, TN = True Negatives, FP = False Positives, FN = False Negatives

            // TODO: write code here to compute the above mentioned variables

            System.out.println("Accuracy="+acc);
            System.out.println("P, R, and F1 score of the positive class=" + p_pos + " " + r_pos + " " + f_pos);
            System.out.println("P, R, and F1 score of the negative class=" + p_neg + " " + r_neg + " " + f_neg);
            System.out.println("Confusion Matrix");
            System.out.println(TP + "\t" + FN);
            System.out.println(FP + "\t" + TN);
        }


        /** Train the Logistic Regression using Stochastic Gradient Ascent **/
        /** Also compute the log-likelihood of the data in this function **/
        public void train(List<LRInstance> instances) {
            for (int n = 0; n < ITERATIONS; n++) {
                double lik = 0.0; // Stores log-likelihood of the training data for this iteration
                for (int i=0; i < instances.size(); i++) {
		    // TODO: Train the model
		    LRInstance currInstance = instances.get(i);
                    // save prob that currInstance's label = 1. It remains the same as we
                    // update the weight vector
		    double probInstance1 = probPred1(currInstance.x); 
		    for(int w = 0; w < weights.length; w++){
                        // VERIFY I DID THIS RIGHT
		        // weight = current + rate*featureval(true label - prob. that this instance's label = 1
			weights[w] = weights[w] + (rate*currInstance.x[w]) * 
                                       (currInstance.label - probInstance1);
		    }
		    // TODO: Compute the log-likelihood of the data here. Remember to take logs when necessary
                    // VERIFY I DID THIS RIGHT
                    // l(W) = true label * (w.x) - log(1 + exp(w.x))
                    double dotProduct = 0;
                    for(int j = 0; j < weights.length; j++){
                        dotProduct += (weights[j] * currInstance.x[j]);
                    }
                    // this uses base e, could do base 2 but would require more math
                    lik = currInstance.label * dotProduct - Math.log(1 + Math.exp(dotProduct));
		}
                System.out.println("iteration: " + n + " lik: " + lik);
            }
        }

        public static class LRInstance {
            public int label; // Label of the instance. Can be 0 or 1
            public double[] x; // The feature vector for the instance

            /** TODO: Constructor for initializing the Instance object **/
            public LRInstance(int label, double[] x) {
            }
        }

        /** Function to read the input dataset **/
        public static List<LRInstance> readDataSet(String file) throws FileNotFoundException {
            List<LRInstance> dataset = new ArrayList<LRInstance>();
            Scanner scanner = null;
            try {
                scanner = new Scanner(new File(file));

                while(scanner.hasNextLine()) {
                    String line = scanner.nextLine();
                    if (line.startsWith("ju")) { // Ignore the header line
                        continue;
                    }
                    String[] columns = line.replace("\n", "").split(",");

                    // every line in the input file represents an instance-label pair
                    int i = 0;
                    double[] data = new double[columns.length - 1];
                    for (i=0; i < columns.length - 1; i++) {
                        data[i] = Double.valueOf(columns[i]);
                    }
                    int label = Integer.parseInt(columns[i]); // last column is the label
                    LRInstance instance = new LRInstance(label, data); // create the instance
                    dataset.add(instance); // add instance to the corpus
                }
            } finally {
                if (scanner != null)
                    scanner.close();
            }
            return dataset;
        }


        public static void main(String... args) throws FileNotFoundException {
            List<LRInstance> trainInstances = readDataSet("HW4_trainset.csv");
            List<LRInstance> testInstances = readDataSet("HW4_testset.csv");

            // create an instance of the classifier
            int d = trainInstances.get(0).x.length;
            LogisticRegression logistic = new LogisticRegression(d);

            logistic.train(trainInstances);

            System.out.println("Norm of the learned weights = "+logistic.weightsL2Norm());
            System.out.println("Length of the weight vector = "+logistic.weights.length);

            // printing accuracy for different values of lambda
            System.out.println("-----------------Printing train set performance-----------------");
            logistic.printPerformance(trainInstances);

            System.out.println("-----------------Printing test set performance-----------------");
            logistic.printPerformance(testInstances);
        }

    }
