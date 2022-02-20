from tkinter import Tk, ttk, Frame, Button, Label, Entry, Text, Checkbutton, Radiobutton,\
     Scale, Listbox, Menu, N, E, S, W, HORIZONTAL, END, FALSE, IntVar, StringVar
#from tkinter import BOTH, RIGHT, RAISED, messagebox as box, PhotoImage
#import os

import numpy as np
import matplotlib.pyplot as plt

import read_text_file_omega as rd
import read_header_line as rhl

import scatterg as a2dscatter
import scatterg3D as a3dscatter
import Normalize_Features as NF
import Randomize_training_samples as RTS
#import Gather_Attrib_NoiseStatistics as SA

#Creating GUI 
class Compact_Frame(Frame):        
     def __init__(self, master):                                 #Initialize the frame
         super(Compact_Frame, self).__init__(master)             #super class Constructor
         self.pack()                                             #Make the frame a pack to control geometry.
         
         self.FeaturesDef = ['X1',
                        'X2',
                        'X3',
                        '...',
                        '...',
                        'Y']
         self.NormalizeVar=0
         self.v=IntVar(self)
         self.v.set(1) #Initialize without normalization or randomization
         self.v_dim=IntVar(self)
         self.v_dim.set(2) #Initialize dimensions to 2D
         self.sizeTrainXval=StringVar(self)
         self.sizeTrainXval.set('Click Button')
         self.centreWindow()
         self.create_widgets()
         self.plot_data()
         self.plot_predictions()
         self.dimension()

     def create_widgets(self):                                 #Create a new class for widgets

         #Event Menu Bar 
         menubar = Menu(self.master)
         self.master.config(menu=menubar)
         fileMenu = Menu(menubar)
         fileMenu.add_command(label="Input Training Data", command=self.input_train_file)
         fileMenu.add_command(label="Input Unlabeled Data", command=self.input_unlabeled_file)
         fileMenu.add_command(label="Exit", command=self.quit)
         menubar.add_cascade(label="File", menu=fileMenu)

         #Event Label/text box that will be shown adjacent text box (Training File Label)
         self.TrainingNameLabel = Label(self, text="Training File:")
         self.TrainingNameLabel.grid(row=0, column=0, sticky=W+E)      
         self.TrainingNameText = Entry(self, width=20)
         self.TrainingNameText.grid(row=0, column=1, padx=5, pady=5, ipady=2, sticky=W+E)

         #Event Label/text box that will be shown adjacent text box (Unlabeled (application File Label)
         self.ApplicationNameLabel = Label(self, text="Application File:")
         self.ApplicationNameLabel.grid(row=1, column=0, sticky=W+E)
         self.ApplicationNameText = Entry(self, width=20)
         self.ApplicationNameText.grid(row=1, column=1, padx=5, pady=5, ipady=2, sticky=W+E)

         #Event Feature List Box
         self.FeaturesDefList = Listbox(self, height=6)
         for t in self.FeaturesDef:
             self.FeaturesDefList.insert(END, t)
         self.FeaturesDefList.grid(row=2, column=0, columnspan=2, pady=5, sticky=N+E+S+W)

         #Event Labels and text boxes to populate matrix
         self.X1_label = Label(self, text="X1 in column")
         self.X1_label.grid(row=4, column=0, sticky=W)
         self.X2_label = Label(self, text="X2 in column")
         self.X2_label.grid(row=5, column=0, sticky=W)
         self.X3_label = Label(self, text="X3 in column")
         self.X3_label.grid(row=6, column=0, sticky=W)
         self.y_label = Label(self, text="Label in column")
         self.y_label.grid(row=7, column=0, sticky=W)
         self.num_labels = Label(self, text="# of Labels")
         self.num_labels.grid(row=8, column=0, sticky=W)
         self.X1 = Entry(self)
         self.X1.grid(row=4, column=1, sticky=W)
         self.X2 = Entry(self)
         self.X2.grid(row=5, column=1, sticky=W)
         self.X3 = Entry(self)
         self.X3.grid(row=6, column=1, sticky=W)
         self.y = Entry(self)
         self.y.grid(row=7, column=1, sticky=W)
         self.num_labels = Entry(self)
         self.num_labels.grid(row=8, column=1, sticky=W)

         #Event icon in to Populate matrix
         from PIL import Image, ImageTk
         self.image = Image.open("LeftArrow_Icon2.JPG")
         self.resized = self.image.resize((23, 10),Image.ANTIALIAS)
         self.arrow=ImageTk.PhotoImage(self.resized)

         #Event buttons to Populate matrix         
         self.X1_button = Button(self, width=10, command=self.X1bttnAction)
         self.X1_button.grid(row=4, column=1, padx=130, sticky=W)
         self.X1_button.config(image=self.arrow, width=25,height=12)
         self.X2_button = Button(self, width=10, command=self.X2bttnAction)
         self.X2_button.grid(row=5, column=1, padx=130, sticky=W)
         self.X2_button.config(image=self.arrow, width=25,height=12)
         self.X3_button = Button(self, width=10, command=self.X3bttnAction)
         self.X3_button.grid(row=6, column=1, padx=130, sticky=W)
         self.X3_button.config(image=self.arrow, width=25,height=12)
         self.y_button = Button(self, width=10, command=self.ybttnAction)
         self.y_button.grid(row=7, column=1, padx=130, sticky=W)
         self.y_button.config(image=self.arrow, width=25,height=12)
         
#########                          
         #Event Matrix Text after selecting the features and averaging
         self.MatrixText = Text(self, padx=5, pady=5, width=20, height=10)
         self.MatrixText.grid(row=9, column=0, columnspan=3, pady=5, sticky=N+E+S+W)

         #Event Label Text after selecting the features and averaging
         self.labelText = Text(self, padx=5, pady=5, width=20, height=10)
         self.labelText.grid(row=9, column=3, columnspan=1, pady=5, sticky=N+E+S+W)

## columns>=2
         #Event Sliding bar Training Samle Size and label          
         self.sizeVar = StringVar()
         self.sizeLabel = Label(self, text="Training Sample Size:",
                             textvariable=self.sizeVar)
         self.sizeLabel.grid(row=0, column=2, columnspan=2, sticky=W+E)
         self.sizeScale = Scale(self, from_=10, to=100, orient=HORIZONTAL,
                             resolution=5, command=self.onsizeScale, tickinterval=10)
         self.sizeScale.grid(row=1, column=2, columnspan=2, padx=10, sticky=W+E)  
         
         #Event label training and xvalidation matrix shapes
#         self.sizeTrainXval=StringVar()
         self.MatrixLabel = Label(self, text="Training Shape:",
                             textvariable=self.sizeTrainXval)
         self.MatrixLabel.grid(row=0, column=4, columnspan=2, padx=10, sticky=E+W)         
         
         #Event create training and xvalidation samples
         self.TrainXvalBtn = Button(self, text="Training Matrix", width=30, command=self.RandomTrainSample)
         self.TrainXvalBtn.grid(row=1, column=4, columnspan=2, padx=10, sticky=E+W) 

         #Event Checkbutton to normalize training data. 
         self.NormalizeVar = IntVar()
         self.NormalizeCheckBtn = Checkbutton(self, text="Normalized?", variable=self.NormalizeVar, command=self.NormChecked)
         self.NormalizeCheckBtn.grid(row=2, column=4, sticky=W+N)   

         #Event estimate Average
         self.AvgBtn = Button(self, text="Gather Average", width=10, command=self.GatherAverage)
         self.AvgBtn.grid(row=2, column=2, padx=5, pady=40, sticky=W+N+S)

         self.check_Matrix = Button(self, text="Define/Check Matrix X", command=self.reveal_matrix)
         self.check_Matrix.grid(row=7, column=3, sticky=W+E)

         #Label Method Selection Combo Box
         self.algorithmLabel = Label(self, text="Method")
         self.algorithmLabel.grid(row=10, column=0, sticky=W+E)

######### Bottom, application of the algorithms
         #Event Combo Box 1 Method of Machine Learning
         self.methodVar = StringVar()
         self.methodCombo = ttk.Combobox(self, textvariable=self.methodVar)
         self.methodCombo['values'] = ('Select Method',
                                       'Naive Bayes',
                                       'Logistical Regression',
                                       'Support Vector Machine',
                                       'Neural Network SciKit',
                                       'Neural Network 2')
         self.methodCombo.current(0)
         self.methodCombo.bind("<<ComboboxSelected>>", self.newMethod)
         self.methodCombo.grid(row=10, column=1, padx=5, pady=10, ipady=4, sticky=W) 

         #Event estimate Average
         self.RunMLalgBtn = Button(self, text="Run ML Alg.", width=10, command=self.RunML)
         self.RunMLalgBtn.grid(row=10, column=2, padx=5, pady=3, sticky=W+E)

         #Output labeled data
         self.outputBtn=Button(self, text="Output", width=10, command=self.outputtxt)
         self.outputBtn.grid(row=9, column=4, padx=5, pady=3, sticky=W+E)          
         
         #Event Score
         self.scoreVar = StringVar()
         self.scoreVar.set("Score")
         
         #Label Score
         self.scoreLabel= Label(self, textvariable=self.scoreVar)
         self.scoreLabel.grid(row=10, column=3, sticky=W+E)

         #Event runtime
         self.runtimeVar = StringVar()
         self.runtimeVar.set("Runtime")
         
         #Label Score
         self.runtimeLabel= Label(self, textvariable=self.runtimeVar)
         self.runtimeLabel.grid(row=10, column=4, sticky=W+E)              

######### Matrix Dimensions
     def dimension(self):          
        options_dim =[("1D",1),("2D",2),("3D",3),("All D",4)]
        for txt_dim,val_dim in options_dim:
            self.options_dim_radiobuttons=Radiobutton(self,
                                              text=txt_dim,
                                              padx=20,
                                              variable=self.v_dim,
                                              value=val_dim)
            self.options_dim_radiobuttons.grid(row=3+val_dim, column=2, sticky=W)
########################

######### Bottom, Plot Left
     def plot_data(self):                                 #Scatter Widgets
         
          self.HistogramBtn = Button(self, text="Histogram", width=10, command=self.HistogramClass)
          self.HistogramBtn.grid(row=11, column=0, padx=5, pady=3, sticky=W+E)

          self.a1dscatterBtn = Button(self, text="1DScatter", width=10, command=self.scatter1D)
          self.a1dscatterBtn.grid(row=12, column=0, padx=5, pady=3, sticky=W+E)
          
          self.a2dscatterBtn = Button(self, text="2DScatter", width=10, command=self.scatter2D)
          self.a2dscatterBtn.grid(row=13, column=0, padx=5, pady=3, sticky=W+E)
          
          self.a3dscatterBtn = Button(self, text="3DScatter", width=10, command=self.scatter3D)
          self.a3dscatterBtn.grid(row=14, column=0, padx=5, pady=3, sticky=W+E)
          
          options =[("Average Gather",1),
                  ("Average Gather Randomized Training",2),
                  ("Average Gather Randomized Xvalidation",3),
                  ("Average Gather Normalized Random Training",4),
                  ("Average Gather Normalized Random Xvalidation",5),
                  ("Full Gather",6),
                  ]
          for txt,val in options:
            self.options_radiobuttons=Radiobutton(self,
                                                  text=txt,
                                                  padx=20,
                                                  variable=self.v,
                                                  value=val)
            self.options_radiobuttons.grid(row=10+val, column=1, sticky=W)

######### Bottom, Plot right

     def plot_predictions(self):                                 #Scatter Widgets
         
          self.HistogramBtn_pred = Button(self, text="HistogramPred", width=10, command=self.HistogramClass_pred)
          self.HistogramBtn_pred.grid(row=11, column=2, padx=5, pady=3, sticky=W+E)

          self.a1dscatterBtn_pred = Button(self, text="1DScatterPred", width=10, command=self.scatter1D_pred)
          self.a1dscatterBtn_pred.grid(row=12, column=2, padx=5, pady=3, sticky=W+E)
          
          self.a2dscatterBtn_pred = Button(self, text="2DScatterPred", width=10, command=self.scatter2D_pred)
          self.a2dscatterBtn_pred.grid(row=13, column=2, padx=5, pady=3, sticky=W+E)
          
          self.a3dscatterBtn_pred = Button(self, text="3DScatterPred", width=10, command=self.scatter3D_pred)
          self.a3dscatterBtn_pred.grid(row=14, column=2, padx=5, pady=3, sticky=W+E)
          
          self.quickplotBtn=Button(self, text="QuickPlot_ppt", width=10, command=self.quickplotppt)
          self.quickplotBtn.grid(row=15, column=2, padx=5, pady=3, sticky=W+E)              
###################### Event Handlers
            
     #Event handler: center the window and size it adequately
     def centreWindow(self):
         w = 1100
         h = 780
         sw = self.master.winfo_screenwidth()
         sh = self.master.winfo_screenheight()
         x = (sw - w)/2
         y = (sh - h)/2
         self.master.geometry('%dx%d+%d+%d' % (w, h, x, y))     

     #Event handler: Input training data via Menu Bar
     def input_train_file(self):
          global X
          global header_list
          from tkinter.filedialog import askopenfilename
          filename=askopenfilename()
          header_list=rhl.read_header_line(filename)
          X=rd.read_text_file(filename)
          self.TrainingNameText.delete(0, END)
          self.TrainingNameText.insert(0,filename)
          self.FeaturesDefList.delete(0, END)
      #Event handler FeaturesDefList
          counter=0          
          for t in header_list:
             header_column=str(counter)
             self.FeaturesDefList.insert(END, t + "     "+ header_column)
             counter=counter+1
          print (X)          

     #Event handler: Input unlabeled application data via Menu Bar          
     def input_unlabeled_file(self):
          global Xapp
          global output_file
          from tkinter.filedialog import askopenfilename
          filename=askopenfilename()
          Xapp=rd.read_text_file(filename)
          self.ApplicationNameText.delete(0, END)
          self.ApplicationNameText.insert(0,filename)
          print (Xapp)
          output_file=filename+"_out.txt"
          
     #Event handler: Button Get Averages    
     def GatherAverage(self):
          global XGather
          global XappGather
#          XGather=SA.XGather(X)
#          print (XGather)
#          XappGather=SA.XGather(Xapp)
          XGather=X
          XappGather=Xapp 

########################################
     #Event Handler, buttons to update matrix          
     def X1bttnAction(self):
          X1feature=self.FeaturesDefList.curselection()
          self.X1.insert(END,X1feature)
          
     def X2bttnAction(self):
          X2feature=self.FeaturesDefList.curselection()
          self.X2.insert(END,X2feature)

     def X3bttnAction(self):
          X3feature=self.FeaturesDefList.curselection()
          self.X3.insert(END,X3feature)
          
     def ybttnAction(self):
          yfeature=self.FeaturesDefList.curselection()
          self.y.insert(END,yfeature)
############ end buttons for ease of matrix definition
                  
     #Event Handler MatrixText
     def reveal_matrix(self):
          import numpy as np
          global Xin, u_label, Xout

          global u_label
          u_label=np.zeros((len(XGather[:,1]),1))
          col_y = self.y.get()
          if col_y == "":
               message = "Always need Label"
               self.y.insert(END,message)
          else:
               col_y=int(col_y)
               u_label=XGather[:,col_y]

          chosen_dim=self.v_dim.get()
          if chosen_dim==1:                  
              Xin=np.zeros((len(XGather[:,1])))[np.newaxis].T
              Xout=np.zeros((len(XappGather[:,1])))[np.newaxis].T
          elif chosen_dim==2:                  
              Xin=np.zeros((len(XGather[:,1]),2))
              Xout=np.zeros((len(XappGather[:,1]),2))
          elif chosen_dim==3:                  
              Xin=np.zeros((len(XGather[:,1]),3))
              Xout=np.zeros((len(XappGather[:,1]),3))
          elif chosen_dim==4:
              Xin=np.delete(XGather,col_y,axis=1)
              Xout=np.delete(XappGather,col_y,axis=1)                           
              
          col_X1 = self.X1.get()
          if col_X1 == "":
               message = "Neeed # for 1D,2D,3D"
               self.X1.insert(END,message)
          else:
               col_X1=int(col_X1)
               Xin[:,0]=XGather[:,col_X1]
               Xout[:,0]=XappGather[:,col_X1]

          col_X2 = self.X2.get()
          if col_X2 == "":
               message = "Neeed # for 2D,3D"
               self.X2.insert(END,message)
          else:
               col_X2=int(col_X2)
               Xin[:,1]=XGather[:,col_X2]
               Xout[:,1]=XappGather[:,col_X2]

          col_X3 = self.X3.get()
          if col_X3 == "":
               message = "Neeed # for 3D"
               self.X3.insert(END,message)
          else:
               col_X3=int(col_X3)
               Xin[:,2]=XGather[:,col_X3]
               Xout[:,2]=XappGather[:,col_X3]             
              
          self.MatrixText.delete(0.0,END)
          self.MatrixText.insert(0.0,Xin[0:9,:])
          self.labelText.delete(0.0,END)
          display_label=np.array(u_label[0:9])[np.newaxis].T
          print(display_label)
          self.labelText.insert(0.0,display_label[0:9,:])

     #Event Handler Button MatrixShape: 
     def RandomTrainSample(self):
         global Xtrain,labeltrain,X_xval,label_xval
         train_size_pct=self.sizeScale.get()
         Xtrain,labeltrain,X_xval,label_xval=RTS.randomTrain(Xin,u_label,train_size_pct)
         self.sizeTrainXval.set("Training Matrix: " + str(Xtrain.shape)+ " XValidation Matrix: " + str(X_xval.shape))
         normalization_choice=self.NormalizeVar.get()
         if normalization_choice==1:
             Xtrain=NF.featureNormalize(Xtrain)
             X_xval=NF.featureNormalize(X_xval)

     #Event handler: Combo Box 1, algrithm
     def newMethod(self, event):
         print(self.methodVar.get())
         algorithm=self.methodCombo.get()
         if algorithm=='Support Vector Machine':
             self.svm_parms() 
         elif algorithm=='Logistical Regression':
             self.logisticReg_parms()
         elif algorithm=='Neural Network SciKit':
             self.NeuralNetSciKit_parms()
         elif algorithm=='Neural Network':
             self.NeuralNet_parms()
         elif algorithm=='Naive Bayes':
             self.NaiveBayes_parms()
                                  
     def svm_parms(self):
         self.kernel_label = Label(self, text="kernel                  ")
         self.kernel_label.grid(row=11, column=3, sticky=W)
         self.kernel_text = Entry(self)
         self.kernel_text.grid(row=11, column=4, sticky=W)
         self.kernel_text.delete(0,END)
         self.kernel_text.insert(0,'rbf')
         self.gamma_label = Label(self, text="Gamma                       ")
         self.gamma_label.grid(row=12, column=3, sticky=W) 
         self.gamma_text = Entry(self)
         self.gamma_text.grid(row=12, column=4, sticky=W)
         self.gamma_text.delete(0,END)
         self.gamma_text.insert(0,'1.0')
         
     def logisticReg_parms(self):
         self.solver_label = Label(self, text="Solver                            ")
         self.solver_label.grid(row=11, column=3, sticky=W)
         self.solver_text = Entry(self)
         self.solver_text.grid(row=11, column=4, sticky=W)
         self.solver_text.delete(0,END)
         self.solver_text.insert(0,'liblinear')
         self.gamma_label = Label(self, text="Inv. Reg Strength")
         self.gamma_label.grid(row=12, column=3, sticky=W) 
         self.gamma_text = Entry(self)
         self.gamma_text.grid(row=12, column=4, sticky=W)
         self.gamma_text.delete(0,END)
         self.gamma_text.insert(0,'1.0')

     def NeuralNet_parms1(self):
         self.numweightsinlayer1_label = Label(self, text="Hidden Layer Size")
         self.numweightsinlayer1_label.grid(row=11, column=3, sticky=W)
         self.numweightsinlayer1_text = Entry(self)
         self.numweightsinlayer1_text.grid(row=11, column=4, sticky=W)
         self.numweightsinlayer1_text.delete(0,END)
         self.numweightsinlayer1_text.insert(0,'25')
         self.maxiter_label = Label(self, text="Maximum Iterations     ")
         self.maxiter_label.grid(row=12, column=3, sticky=W) 
         self.maxiter_text = Entry(self)
         self.maxiter_text.grid(row=12, column=4, sticky=W)   
         self.maxiter_text.delete(0,END)
         self.maxiter_text.insert(0,'250')         
         
     def NeuralNetSciKit_parms(self):
         self.hidden_layer_sizes_label = Label(self, text="Hidden Layer Sizes")
         self.hidden_layer_sizes_label.grid(row=11, column=3, sticky=W)
         self.hidden_layer_sizes_text = Entry(self)
         self.hidden_layer_sizes_text.grid(row=11, column=4, sticky=W)
         self.hidden_layer_sizes_text.delete(0,END)
         self.hidden_layer_sizes_text.insert(0,'5,2')
         self.solver_label = Label(self, text="Solver             ")
         self.solver_label.grid(row=12, column=3, sticky=W) 
         self.solver_text = Entry(self)
         self.solver_text.grid(row=12, column=4, sticky=W)   
         self.solver_text.delete(0,END)
         self.solver_text.insert(0,'lbfgs')
         self.alpha_label = Label(self, text="Alpha             ")
         self.alpha_label.grid(row=13, column=3, sticky=W) 
         self.alpha_text = Entry(self)
         self.alpha_text.grid(row=13, column=4, sticky=W)   
         self.alpha_text.delete(0,END)
         self.alpha_text.insert(0,'1e-5')    
         
     def NaiveBayes_parms(self):
#         self.priors_label = Label(self, text="Priors                         ")
#         self.priors_label.grid(row=11, column=3, sticky=W)
#         self.priors_text = Entry(self)
#         self.priors_text.grid(row=11, column=4, sticky=W)
#         self.priors_text.delete(0,END)
#         self.priors_text.insert(0,'None')
         self.priorsdesc_label = Label(self, text="                                       ")
         self.priorsdesc_label.grid(row=11, column=3, sticky=W) 
         self.priorsdesc2_label = Label(self, text="                                       ")
         self.priorsdesc2_label.grid(row=11, column=4, sticky=W) 
         self.priorsdesc_label = Label(self, text="                                       ")
         self.priorsdesc_label.grid(row=12, column=3, sticky=W) 
         self.priorsdesc2_label = Label(self, text="                                       ")
         self.priorsdesc2_label.grid(row=12, column=4, sticky=W) 
         
     #Event Handler: 
     def RunML(self):
         from time import time
         global Xout_pred_label,train_pred_label,xval_pred_label
         print ("Xin:", Xin.shape, "Xtrain:",Xtrain.shape, "ytrain: ", labeltrain.size, "Xxval: ",X_xval.shape,"yval: ",label_xval.shape)

         normalization_choice=self.NormalizeVar.get()
         Xout_pred=Xout
         if normalization_choice==1:
             Xout_pred=NF.featureNormalize(Xout)
         print("Normalization (0=False, 1=True): ",normalization_choice)    

         algorithm=self.methodCombo.get()
         print("Machine Learning Algorithm chosen: ",algorithm)
         
         t0 = time()
         
         if algorithm=='Naive Bayes':
              from sklearn.naive_bayes import GaussianNB
#              priors_string=self.priors_text.get()
#              classify=GaussianNB(priors=priors_string)
              classify=GaussianNB()
              classify.fit(Xtrain,labeltrain)
              accuracy=classify.score(X_xval, label_xval)
              self.scoreVar.set(accuracy)
              train_pred_label=classify.predict(Xtrain)
              xval_pred_label=classify.predict(X_xval)
              Xout_pred_label=classify.predict(Xout_pred)
              print("accuracy= ",accuracy)              
         elif algorithm=='Logistical Regression':
              from sklearn import linear_model                         
              solver = self.solver_text.get()
              Cstring=self.gamma_text.get()
              C=float(Cstring)  
              classify=linear_model.LogisticRegression(solver=solver,C=C)
              classify.fit(Xtrain,labeltrain)
              accuracy=classify.score(X_xval, label_xval)
              self.scoreVar.set(accuracy)
              print("Solver= ",solver,"C= ",C,"accuracy= ",accuracy)
              train_pred_label=classify.predict(Xtrain)
              xval_pred_label=classify.predict(X_xval)
              Xout_pred_label=classify.predict(Xout_pred)
         elif algorithm=='Support Vector Machine':
              from sklearn import svm
              svm_kernel = self.kernel_text.get()
              Cstring=self.gamma_text.get()
              C=float(Cstring)                            
              classify=svm.SVC(kernel=svm_kernel,C=C)
              classify.fit(Xtrain,labeltrain)
              accuracy=classify.score(X_xval, label_xval)
              self.scoreVar.set(accuracy)
              print("kernel= ",svm_kernel,"C= ",C, "accuracy= ",accuracy)
              train_pred_label=classify.predict(Xtrain)
              xval_pred_label=classify.predict(X_xval)
              Xout_pred_label=classify.predict(Xout_pred)
         elif algorithm=='Neural Network SciKit':
              from sklearn.neural_network import MLPClassifier
              solver = self.solver_text.get()
              hidden_layer_sizes=self.hidden_layer_sizes_text.get()
              hidden_layer_sizes=np.fromstring(hidden_layer_sizes, dtype=int, sep=',')
              alpha = self.alpha_text.get()
              alpha = int(alpha)
              classify = MLPClassifier(solver=solver, alpha=alpha, hidden_layer_sizes=(hidden_layer_sizes), random_state=1)
              classify.fit(Xtrain,labeltrain)
              accuracy=classify.score(X_xval, label_xval)
              self.scoreVar.set(accuracy)
              print("Size of Hidden Layers= ",hidden_layer_sizes,"Solver= ",solver,"Alpha= ",alpha,"accuracy= ",accuracy)
              train_pred_label=classify.predict(Xtrain)
              xval_pred_label=classify.predict(X_xval)
              Xout_pred_label=classify.predict(Xout_pred)
         elif algorithm=='Neural Network 2':
              import NeuronalNetwork2 as nn
              hidden_size_string=self.numweightsinlayer1_text.get()
              hidden_size=int(hidden_size_string)  
#              hidden_size = 25
              learning_rate = 1
              max_iter_string=self.maxiter_text.get()
              max_iter=int(max_iter_string)  
#              max_iter=250
              num_labels=int(self.num_labels.get())
              nntraining_weights=nn.neuronal_network(Xtrain,labeltrain,num_labels,hidden_size,learning_rate,max_iter)
              xval_pred_label=nn.neuronal_network_predict(X_xval,nntraining_weights,hidden_size,num_labels)
              Xout_pred_label=nn.neuronal_network_predict(Xout_pred,nntraining_weights,hidden_size,num_labels)
              accuracy=nn.score(label_xval,xval_pred_label)
              self.scoreVar.set(accuracy)
              print("Hidden Layer SIze: ",hidden_size, " Maximum Iterations",max_iter, "accuracy= ",accuracy)
         else:
               print('not a valid algorithm')
         
         tend=time()
         runtime=tend-t0
         runtimestring="T&P", round(runtime, 3), "s"
         print (runtimestring)
         self.runtimeVar.set(runtimestring)

     #Event handler: Sliding Bar Label, Training and xvalidation Sample Size
     def onsizeScale(self, val):
         self.sizeVar.set("Training Size: " + str(val) + "%"+"  Xvalidation Size: " + str(100-int(val)) + "%")    

     #Event handler: Normalize data
     def NormChecked(self):
         if self.NormalizeVar.get() == 1:
             self.master.title("Features Normalized")
         else:
             self.master.title("Features Raw")     

     #Event handler: Normalize data
     def newTitle(self, val):
         sender = val.widget
         idx = sender.curselection()
         value = sender.get(idx)
         self.scoreVar.set(value)         

#########Plot Input
     #Event Handler, option data plot
     def defXfit(self):
         global Xfit, fit_label
         chosen=self.v.get()
         if chosen==1:
             Xfit=Xin
             fit_label=u_label
             print ("Gather avg")
         elif chosen==2:
             Xfit=Xtrain
             fit_label=labeltrain
             print ("rand train")
         elif chosen==3:
             Xfit=X_xval
             fit_label=label_xval
             print ("rand xval")
         elif chosen==4:
             Xfit=NF.featureNormalize(Xtrain)
             fit_label=labeltrain
             print ("norm rand train")
         elif chosen==5:
             Xfit=NF.featureNormalize(X_xval)
             fit_label=label_xval
             print ("norm rand xval")
         else:
             print ("option is not valid")

     #Event Handler, 1D Scatter Button
     def HistogramClass(self):
          import matplotlib.pyplot as plt
          self.defXfit()
          number_of_labels=int(self.num_labels.get())
          bins=np.arange(number_of_labels+1)-0.5
          print(bins,fit_label)      
          plt.hist(fit_label, bins)
          plt.title('Histogram Input')
          plt.xlabel('Label')
          plt.ylabel('Normalized Dist')
          plt.show()

     #Event Handler, 1D Scatter Button
     def scatter1D(self):
          self.defXfit()
          x1=Xfit[:,0]
          x2=fit_label
          x1_axis=header_list[int(self.X1.get())]
          number_of_labels=int(self.num_labels.get())
          a2dscatter.scatterg(x1,x2,fit_label,number_of_labels,x1_axis,'LABEL','1D Input Label')
          plt.show()
          
     #Event Handler, 2D Scatter Button
     def scatter2D(self):
          self.defXfit()
          x1=Xfit[:,0]
          x2=Xfit[:,1]
          x1_axis=header_list[int(self.X1.get())]
          x2_axis=header_list[int(self.X2.get())]
          number_of_labels=int(self.num_labels.get())
          a2dscatter.scatterg(x1,x2,fit_label,number_of_labels,x1_axis,x2_axis, '2D Input Label')           
          plt.show()
          
     #Event Handler, 3D Scatter Button
     def scatter3D(self):
          self.defXfit()
          x1=Xfit[:,0]
          x2=Xfit[:,1]
          x3=Xfit[:,2]
          x1_axis=header_list[int(self.X1.get())]
          x2_axis=header_list[int(self.X2.get())]
          x3_axis=header_list[int(self.X3.get())]
          number_of_labels=int(self.num_labels.get())
          a3dscatter.scatterg(x1,x2,x3,fit_label,number_of_labels,x1_axis,x2_axis,x3_axis, '3D Input Label')
          plt.show()
          
     #Event Handler, Output Text files with appended Label
     def outputtxt(self):
          import numpy as np
          print("matrix shape",XappGather.shape)
          dim2_Xout_pred_label=np.array(Xout_pred_label)[np.newaxis].T
          print("pred label shape",dim2_Xout_pred_label.shape)
          Xappend=np.append(XappGather,dim2_Xout_pred_label,axis=1)
          print("Output matrix shape",Xappend.shape)
          np.savetxt(output_file, Xappend)

#########Plot Output Predictions
     #Event Handler, option data plot
     def defXpred(self):
         global Xpred_plot, pred_label_plot
         chosen=self.v.get()
         if chosen==1:
             Xpred_plot=Xout
             pred_label_plot=Xout_pred_label
             print ("Gather avg")
         elif chosen==2:
             Xpred_plot=Xtrain
             pred_label_plot=train_pred_label
             print ("rand train")
         elif chosen==3:
             Xpred_plot=X_xval
             pred_label_plot=xval_pred_label
             print ("rand xval")
         elif chosen==4:
             Xpred_plot=NF.featureNormalize(Xtrain)
             pred_label_plot=labeltrain
             print ("norm rand train")
         elif chosen==5:
             Xpred_plot=NF.featureNormalize(X_xval)
             pred_label_plot=label_xval
             print ("norm rand xval")
         else:
             print ("option is not valid")

     #Event Handler, 1D Scatter Button
     def HistogramClass_pred(self):
          import matplotlib.pyplot as plt
          self.defXpred()
          number_of_labels=int(self.num_labels.get())
          bins=np.arange(number_of_labels+1)-0.5        
          plt.hist(pred_label_plot, bins)
          plt.title('Histogram Predicted')
          plt.show()

     #Event Handler, 1D Scatter Button
     def scatter1D_pred(self):
          self.defXpred()
          x1=Xpred_plot[:,0]
          x2=pred_label_plot
          x1_axis=header_list[int(self.X1.get())]
          number_of_labels=int(self.num_labels.get())
          a2dscatter.scatterg(x1,x2,pred_label_plot,number_of_labels,x1_axis,'LABEL','1D Predicted')
          plt.show()
          
     #Event Handler, 2D Scatter Button
     def scatter2D_pred(self):
          self.defXpred()
          x1=Xpred_plot[:,0]
          x2=Xpred_plot[:,1]
          x1_axis=header_list[int(self.X1.get())]
          x2_axis=header_list[int(self.X2.get())]
          number_of_labels=int(self.num_labels.get())
          a2dscatter.scatterg(x1,x2,pred_label_plot,number_of_labels,x1_axis,x2_axis,'2D Predicted')
          plt.show()
          
     #Event Handler, 3D Scatter Button
     def scatter3D_pred(self):
          self.defXpred()
          x1=Xpred_plot[:,0]
          x2=Xpred_plot[:,1]
          x3=Xpred_plot[:,2]
          x1_axis=header_list[int(self.X1.get())]
          x2_axis=header_list[int(self.X2.get())]
          x3_axis=header_list[int(self.X3.get())]
          number_of_labels=int(self.num_labels.get())
          a3dscatter.scatterg(x1,x2,x3,pred_label_plot,number_of_labels,x1_axis,x2_axis,x3_axis, '3D Predicted')
          plt.show()
          
     #Event Handler, Quick Plot for ppt
     def quickplotppt(self):
          import matplotlib.pyplot as plt
          
          self.defXfit()
          self.defXpred()
          
          # Histograms         
          number_of_labels=int(self.num_labels.get())
          bins=np.arange(number_of_labels+1)-0.5     
          plt.hist(fit_label, bins, density=True)
          plt.title('Histogram Input')
          plt.show()

          number_of_labels=int(self.num_labels.get())
          bins=np.arange(number_of_labels+1)-0.5     
          plt.hist(pred_label_plot, bins, density=True)
          plt.title('Histogram Predicted')
          plt.show()

          # 1D Scatters
          x1=Xfit[:,0]
          x2=fit_label
          x1_axis=header_list[int(self.X1.get())]
          number_of_labels=int(self.num_labels.get())
          a2dscatter.scatterg(x1,x2,fit_label,number_of_labels,x1_axis,'LABEL','1D Input Label')
          plt.show()
          
          x1=Xpred_plot[:,0]
          x2=pred_label_plot
          x1_axis=header_list[int(self.X1.get())]
          number_of_labels=int(self.num_labels.get())
          a2dscatter.scatterg(x1,x2,pred_label_plot,number_of_labels,x1_axis,'LABEL','1D Predicted')
          plt.show()
          
          # 2D Scatters
          x1=Xfit[:,0]
          x2=Xfit[:,1]
          x1_axis=header_list[int(self.X1.get())]
          x2_axis=header_list[int(self.X2.get())]
          number_of_labels=int(self.num_labels.get())
          a2dscatter.scatterg(x1,x2,fit_label,number_of_labels,x1_axis,x2_axis, '2D Input Label') 
          plt.show()
          
          x1=Xpred_plot[:,0]
          x2=Xpred_plot[:,1]
          x1_axis=header_list[int(self.X1.get())]
          x2_axis=header_list[int(self.X2.get())]
          number_of_labels=int(self.num_labels.get())
          a2dscatter.scatterg(x1,x2,pred_label_plot,number_of_labels,x1_axis,x2_axis, '2D Predicted') 
          plt.show()
          
          # 3D Scatters
          x1=Xfit[:,0]
          x2=Xfit[:,1]
          x3=Xfit[:,2]
          x1_axis=header_list[int(self.X1.get())]
          x2_axis=header_list[int(self.X2.get())]
          x3_axis=header_list[int(self.X3.get())]
          number_of_labels=int(self.num_labels.get())
          a3dscatter.scatterg(x1,x2,x3,fit_label,number_of_labels,x1_axis,x2_axis,x3_axis,'3D Input Label')
          plt.show()
          
          x1=Xpred_plot[:,0]
          x2=Xpred_plot[:,1]
          x3=Xpred_plot[:,2]
          x1_axis=header_list[int(self.X1.get())]
          x2_axis=header_list[int(self.X2.get())]
          x3_axis=header_list[int(self.X3.get())]
          number_of_labels=int(self.num_labels.get())
          a3dscatter.scatterg(x1,x2,x3,pred_label_plot,number_of_labels,x1_axis,x2_axis,x3_axis, '3D Predicted')
          plt.show()
          
def main():
     root = Tk()
     #root.geometry("250x150+300+300")    # width x height + x + y
     # we will use centreWindow instead

     root.resizable(width=FALSE, height=FALSE)
     # .. not resizable

     app = Compact_Frame(root)
     root.mainloop()

if __name__ == '__main__':
     main()
