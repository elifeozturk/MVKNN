// Copyright Header - Multi-View K-Nearest Neighbors (MVKNN)
// Copyright (C) 2020 Elife OZTURK KIYAK <elife.ozturk@ceng.deu.edu.tr>

// Copyright is owned by the author working at Dokuz Eylul University. 
// You can use the algorithm for academic and research purposes only, e.g. not for commercial use, without a fee. 

using System;
using System.IO;
using System.Collections.Generic;
using weka.core;
using weka.classifiers;
using weka.classifiers.meta;
using weka.classifiers.lazy;
using weka.classifiers.evaluation.output.prediction;

namespace MVKNN
{
    class Program
    {
         static int numberofviews = 2;     // The number of views
  
        static List<int>[] predicted = new List<int>[numberofviews + 1];
        static List<int> actual = new List<int>();
  
        // View-Based Classification 
        public static void ViewBasedClassification(Instances insts, int folds, int view, int k)
        {
            java.util.Random rand = new java.util.Random(1);

            insts.setClassIndex(insts.numAttributes() - 1);

            Vote vote = new Vote();
            Classifier[] classifiers = new Classifier[k];

            for (int i = 1; i <= k; i++)
            {
                IBk IBk = new IBk();    // Instance-based (IB) learner
                IBk.setKNN(i);
                classifiers[i - 1] = IBk;
            }

            vote.setClassifiers(classifiers);
            vote.setCombinationRule(new SelectedTag(Vote.MAJORITY_VOTING_RULE, Vote.TAGS_RULES));
            vote.buildClassifier(insts);      // Generate classifiers

            AbstractOutput output = new PlainText();
            output.setBuffer(new java.lang.StringBuffer());
            output.setHeader(insts);

            Evaluation evaluation = new Evaluation(insts);          
            evaluation.crossValidateModel(vote, insts, folds, rand, output);  // n-fold cross validation 
            
            // Parse the output string
            string str = output.getBuffer().toString();
            str = str.Replace("  ", " ").Replace("  ", " ").Replace("  ", " ").Replace("  ", " ");
            string[] line = str.Split('\n');

            predicted[view] = new List<int>();
            predicted[view].Clear();
            actual.Clear();

            for (int i = 1; i < line.Length - 1; i++)
            {
                String[] linesplit = line[i].Trim().Split(' ');

                int index = linesplit[1].IndexOf(':');    // Actual class
                actual.Add(Convert.ToInt16(linesplit[1].Substring(0, index)) - 1);

                index = linesplit[2].IndexOf(':');        // Predicted class
                predicted[view].Add(Convert.ToInt16(linesplit[2].Substring(0, index)) - 1);
            }

        }

        // Multi-View Based Classification
        public static double MultiViewBasedClassification(int numberofinstances, int numberofclasses)
        {
            int count = 0;  // The number of correctly predicted instances

            for (int i = 0; i < numberofinstances; i++)
            {
                int[] classes = new int[numberofclasses];     
                for (int j = 1; j <= numberofviews; j++)  // Counting the prediction of each view for each class 
                {
                    int result = predicted[j][i];
                    classes[result] = classes[result] + 1;        
                }

                int decision = 0; 
                double max = 0;

                for (int j = 0; j < classes.Length; j++)  // Majority voting
                {
                    if (classes[j] >= max)  
                    {
                        max = classes[j];
                        decision = j;
                    }
                }

                if (actual[i] == decision)  // Comparison of the predicted class and actual class
                    count++;
            }
            return Math.Round((((double)count / numberofinstances) * 100), 2);
        }

        static string[] filenames = { "", "phone", "watch" };   // The views of the WISDM dataset
 
        static void Main(string[] args)
        {
            Instances insts = new Instances(new java.io.FileReader("datasets\\" + filenames[1] + ".arff")); // Read the first view data
            
            int folds = 10;
   
            for (int i = 1; i <= numberofviews; i++)
            {
                insts = new Instances(new java.io.FileReader("datasets\\" + filenames[i] + ".arff"));  // Read the view data

                int k = (int) Math.Sqrt(insts.numInstances()) ;   // The number of nearest neighbors            
                ViewBasedClassification(insts, folds, i, k);     // View-based classification
            }

            double accuracy = MultiViewBasedClassification(insts.numInstances(), insts.numClasses()); // Multi-view-based classification 

            Console.WriteLine("Multi-View K-Nearest Neighbors (MVKNN) \n");
            Console.WriteLine("Accuracy = " +  Math.Round(accuracy, 2));

            Console.ReadLine();
        }
    }
}