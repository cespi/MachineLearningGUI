# MachineLearningGUI
An interface to test the performance of different Seiskit Machine Learning algorithm (run time versus training and test score)
My main goal is showing an area of interest. I developed this several years ago, after taking Dr. Ng's training available from coursera. Since then, I have taken more advanced courses and developed new skills. I am currently transferring this code to Javascript html, the tkinter seems to limited. However, this could still be a good starting point to develop and app that can be made portable with Docker. 
How to run:
1) Input the Training (labeled data set)
2) Input the application data set, a sample which has not been labeled for which you want to predict the label
3) The initial intention was havig a level of flexibility to allow for obtaining attributes in a gather or collection ( identifiable by column 1, sort of an index column. For example, using the average age of the alumni as opposed to each student seaprately. This is activated via gather average click, but it is currently commented. It needs to become a more sophisticated "calculator" and then some intricacies come fromexcluding that index column.
4) Pull in the name of the feature into the columns x1, x2, x3 and the labels, by clicking on the arrows or writing the column number shown on the text field.
5) Provide the numer of labels. For example, if the labels are blue/red/yellow/green then use 4, if 0/1 then use two. This part is for visualization purposes and can be changed for different 3D or 2D visualizations via scatter plots. The histograms are limited to the labels only, for now. This needs to change and functionalities as xcorrelations or those from Random Forrest need be added to understand the input.
6) Move the sliding bar to divide the training examples into a training and a cross validation dataset. This will allow obtaining two scores, the training score and the xvalidation, with examples that were not used to create the predictive model. 
7)  It is recomended to use normaliation to improve performance. click on that option on the box.
8)  You may choose to activate every column to use all features. At the moment, there is no option to select more than just 3 input feature and less than all. You woud need to cut the number of columns externally. This is going to improve, the whole input methodology replaced with Pandas libraries and additional ETL functionalities.
9)  Use the method menu to selct the Sklearn algorithm and click on Run ML Alg to measure its performance and run time. 
10) Refer to the pictures for examples
![Run with example file](ExampleRun.png "Screen Grab1")

Histograms
![Histogram Training](Histogram_Traininglabels.png "Hist Train")
![Histogram Predicted](Histogram_PredictedLabels.png "Hist Predicted")

3D Scatters
![3D Training](3D_training.png "3D Scatter Train")
![3D Predicted](3Dpredicted.png "3D Scatter Predicted")
