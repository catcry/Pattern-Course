Python 3.6.7 (v3.6.7:6ec5cf24b7, Oct 20 2018, 13:35:33) [MSC v.1900 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> 
====== RESTART: C:\Users\catcry\Desktop\Pattern\Project\pro\svmpoly.py ======
Traceback (most recent call last):
  File "C:\Users\catcry\Desktop\Pattern\Project\pro\svmpoly.py", line 239, in <module>
    clf = NuSVC(kernel = 'rbf', C=2000, cache_size = 2500, gamma = 'scale')
TypeError: __init__() got an unexpected keyword argument 'C'
>>> 
====== RESTART: C:\Users\catcry\Desktop\Pattern\Project\pro\svmpoly.py ======
======================Results of KDDTrain+ Set: ===================================

The Specification of Classifier is: SVC(C=2000, cache_size=2500, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
Number of un-Detected Attacks (FN) in Training Set is: FN =  235  out of 125973
Number of False Attack Alarms (FP) in Training Set is: FP = 175  out of 125973
Accuracy on Trainin Set: Accuracy =  99.67453343176712 %

======================Results of KDDTest+ Set: ===================================

Number of un-Detected Attacks (FN) in KDDTest+ Set is: FN = 3714  out of 22543
Number of False Attack Alarms (FP) in KDDTest+ Set is: FP = 338  out of 22543
Accuracy on Trainin Set: Accuracy =  82.02546244954088 %

======================Results of KDDTest-21 Set: ===================================

Number of un-Detected Attacks (FN) in Test21 Set is: FN =  3702  out of 11850
Number of False Attack Alarms (FP) in Test21 Set is: FP = 324  out of 11850
Accuracy on Trainin Set: Accuracy =  66.0253164556962 %
>>> 
====== RESTART: C:\Users\catcry\Desktop\Pattern\Project\pro\svmpoly.py ======
Traceback (most recent call last):
  File "C:\Users\catcry\Desktop\Pattern\Project\pro\svmpoly.py", line 241, in <module>
    clf.fit(b, label_train)
  File "C:\Program Files\Python36\lib\site-packages\sklearn\svm\base.py", line 212, in fit
    fit(X, y, sample_weight, solver_type, kernel, random_seed=seed)
  File "C:\Program Files\Python36\lib\site-packages\sklearn\svm\base.py", line 271, in _dense_fit
    max_iter=self.max_iter, random_seed=random_seed)
KeyboardInterrupt
>>> 
====== RESTART: C:\Users\catcry\Desktop\Pattern\Project\pro\svmpoly.py ======
======================Results of KDDTrain+ Set: ===================================

The Specification of Classifier is: SVC(C=1, cache_size=5000, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
Number of un-Detected Attacks (FN) in Training Set is: FN =  2288  out of 125973
Number of False Attack Alarms (FP) in Training Set is: FP = 525  out of 125973
Accuracy on Trainin Set: Accuracy =  97.76698181356322 %

======================Results of KDDTest+ Set: ===================================

Number of un-Detected Attacks (FN) in KDDTest+ Set is: FN = 4912  out of 22543
Number of False Attack Alarms (FP) in KDDTest+ Set is: FP = 247  out of 22543
Accuracy on Trainin Set: Accuracy =  77.11484718094309 %

======================Results of KDDTest-21 Set: ===================================

Number of un-Detected Attacks (FN) in Test21 Set is: FN =  4911  out of 11850
Number of False Attack Alarms (FP) in Test21 Set is: FP = 230  out of 11850
Accuracy on Trainin Set: Accuracy =  56.616033755274266 %
>>> 
====== RESTART: C:\Users\catcry\Desktop\Pattern\Project\pro\svmpoly.py ======
======================Results of KDDTrain+ Set: ===================================

The Specification of Classifier is: SVC(C=10, cache_size=5000, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
Number of un-Detected Attacks (FN) in Training Set is: FN =  1077  out of 125973
Number of False Attack Alarms (FP) in Training Set is: FP = 667  out of 125973
Accuracy on Trainin Set: Accuracy =  98.61557635366309 %

======================Results of KDDTest+ Set: ===================================

Number of un-Detected Attacks (FN) in KDDTest+ Set is: FN = 4378  out of 22543
Number of False Attack Alarms (FP) in KDDTest+ Set is: FP = 230  out of 22543
Accuracy on Trainin Set: Accuracy =  79.55906489819456 %

======================Results of KDDTest-21 Set: ===================================

Number of un-Detected Attacks (FN) in Test21 Set is: FN =  4393  out of 11850
Number of False Attack Alarms (FP) in Test21 Set is: FP = 214  out of 11850
Accuracy on Trainin Set: Accuracy =  61.12236286919831 %
>>> 
====== RESTART: C:\Users\catcry\Desktop\Pattern\Project\pro\svmpoly.py ======
======================Results of KDDTrain+ Set: ===================================

The Specification of Classifier is: SVC(C=100, cache_size=5000, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
Number of un-Detected Attacks (FN) in Training Set is: FN =  404  out of 125973
Number of False Attack Alarms (FP) in Training Set is: FP = 397  out of 125973
Accuracy on Trainin Set: Accuracy =  99.36414946059871 %

======================Results of KDDTest+ Set: ===================================

Number of un-Detected Attacks (FN) in KDDTest+ Set is: FN = 4066  out of 22543
Number of False Attack Alarms (FP) in KDDTest+ Set is: FP = 234  out of 22543
Accuracy on Trainin Set: Accuracy =  80.92534267843676 %

======================Results of KDDTest-21 Set: ===================================

Number of un-Detected Attacks (FN) in Test21 Set is: FN =  4057  out of 11850
Number of False Attack Alarms (FP) in Test21 Set is: FP = 223  out of 11850
Accuracy on Trainin Set: Accuracy =  63.88185654008438 %
>>> 
====== RESTART: C:\Users\catcry\Desktop\Pattern\Project\pro\svmpoly.py ======
======================Results of KDDTrain+ Set: ===================================

The Specification of Classifier is: SVC(C=1000, cache_size=5000, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
Number of un-Detected Attacks (FN) in Training Set is: FN =  269  out of 125973
Number of False Attack Alarms (FP) in Training Set is: FP = 187  out of 125973
Accuracy on Trainin Set: Accuracy =  99.6380176704532 %

======================Results of KDDTest+ Set: ===================================

Number of un-Detected Attacks (FN) in KDDTest+ Set is: FN = 3814  out of 22543
Number of False Attack Alarms (FP) in KDDTest+ Set is: FP = 314  out of 22543
Accuracy on Trainin Set: Accuracy =  81.6883289712993 %

======================Results of KDDTest-21 Set: ===================================

Number of un-Detected Attacks (FN) in Test21 Set is: FN =  3805  out of 11850
Number of False Attack Alarms (FP) in Test21 Set is: FP = 300  out of 11850
Accuracy on Trainin Set: Accuracy =  65.35864978902954 %
>>> 
====== RESTART: C:\Users\catcry\Desktop\Pattern\Project\pro\svmpoly.py ======
======================Results of KDDTrain+ Set: ===================================

The Specification of Classifier is: SVC(C=5000, cache_size=5000, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
Number of un-Detected Attacks (FN) in Training Set is: FN =  212  out of 125973
Number of False Attack Alarms (FP) in Training Set is: FP = 159  out of 125973
Accuracy on Trainin Set: Accuracy =  99.70549244679415 %

======================Results of KDDTest+ Set: ===================================

Number of un-Detected Attacks (FN) in KDDTest+ Set is: FN = 3699  out of 22543
Number of False Attack Alarms (FP) in KDDTest+ Set is: FP = 374  out of 22543
Accuracy on Trainin Set: Accuracy =  81.93230714634254 %

======================Results of KDDTest-21 Set: ===================================

Number of un-Detected Attacks (FN) in Test21 Set is: FN =  3704  out of 11850
Number of False Attack Alarms (FP) in Test21 Set is: FP = 362  out of 11850
Accuracy on Trainin Set: Accuracy =  65.68776371308016 %
>>> 
====== RESTART: C:\Users\catcry\Desktop\Pattern\Project\pro\svmpoly.py ======
======================Results of KDDTrain+ Set: ===================================

The Specification of Classifier is: NuSVC(cache_size=5000, class_weight=None, coef0=0.0,
   decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
   max_iter=-1, nu=0.1, probability=False, random_state=None,
   shrinking=True, tol=0.001, verbose=False)
Number of un-Detected Attacks (FN) in Training Set is: FN =  3462  out of 125973
Number of False Attack Alarms (FP) in Training Set is: FP = 534  out of 125973
Accuracy on Trainin Set: Accuracy =  96.82789169107666 %

======================Results of KDDTest+ Set: ===================================

Number of un-Detected Attacks (FN) in KDDTest+ Set is: FN = 5109  out of 22543
Number of False Attack Alarms (FP) in KDDTest+ Set is: FP = 188  out of 22543
Accuracy on Trainin Set: Accuracy =  76.50268375992547 %

======================Results of KDDTest-21 Set: ===================================

Number of un-Detected Attacks (FN) in Test21 Set is: FN =  5102  out of 11850
Number of False Attack Alarms (FP) in Test21 Set is: FP = 175  out of 11850
Accuracy on Trainin Set: Accuracy =  55.46835443037974 %
>>> 
====== RESTART: C:\Users\catcry\Desktop\Pattern\Project\pro\svmpoly.py ======
======================Results of KDDTrain+ Set: ===================================

The Specification of Classifier is: SVC(C=5000, cache_size=5000, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
Number of un-Detected Attacks (FN) in Training Set is: FN =  285  out of 125973
Number of False Attack Alarms (FP) in Training Set is: FP = 314  out of 125973
Accuracy on Trainin Set: Accuracy =  99.52450128202075 %

======================Results of KDDTest+ Set: ===================================

Number of un-Detected Attacks (FN) in KDDTest+ Set is: FN = 3774  out of 22543
Number of False Attack Alarms (FP) in KDDTest+ Set is: FP = 326  out of 22543
Accuracy on Trainin Set: Accuracy =  81.8125360422304 %

======================Results of KDDTest-21 Set: ===================================

Number of un-Detected Attacks (FN) in Test21 Set is: FN =  3763  out of 11850
Number of False Attack Alarms (FP) in Test21 Set is: FP = 311  out of 11850
Accuracy on Trainin Set: Accuracy =  65.62025316455696 %
>>> 
====== RESTART: C:\Users\catcry\Desktop\Pattern\Project\pro\svmpoly.py ======
======================Results of KDDTrain+ Set: ===================================

The Specification of Classifier is: SVC(C=2000, cache_size=5000, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
Number of un-Detected Attacks (FN) in Training Set is: FN =  235  out of 125973
Number of False Attack Alarms (FP) in Training Set is: FP = 175  out of 125973
Accuracy on Trainin Set: Accuracy =  99.67453343176712 %

======================Results of KDDTest+ Set: ===================================

Number of un-Detected Attacks (FN) in KDDTest+ Set is: FN = 3714  out of 22543
Number of False Attack Alarms (FP) in KDDTest+ Set is: FP = 338  out of 22543
Accuracy on Trainin Set: Accuracy =  82.02546244954088 %

======================Results of KDDTest-21 Set: ===================================

Number of un-Detected Attacks (FN) in Test21 Set is: FN =  3702  out of 11850
Number of False Attack Alarms (FP) in Test21 Set is: FP = 324  out of 11850
Accuracy on Trainin Set: Accuracy =  66.0253164556962 %
>>> 
====== RESTART: C:\Users\catcry\Desktop\Pattern\Project\pro\svmpoly.py ======
Traceback (most recent call last):
  File "C:\Users\catcry\Desktop\Pattern\Project\pro\svmpoly.py", line 241, in <module>
    clf.fit(b, label_train)
  File "C:\Program Files\Python36\lib\site-packages\sklearn\svm\base.py", line 212, in fit
    fit(X, y, sample_weight, solver_type, kernel, random_seed=seed)
  File "C:\Program Files\Python36\lib\site-packages\sklearn\svm\base.py", line 271, in _dense_fit
    max_iter=self.max_iter, random_seed=random_seed)
  File "sklearn\svm\libsvm.pyx", line 187, in sklearn.svm.libsvm.fit
ValueError: specified nu is infeasible
>>> 
====== RESTART: C:\Users\catcry\Desktop\Pattern\Project\pro\svmpoly.py ======
======================Results of KDDTrain+ Set: ===================================

The Specification of Classifier is: NuSVC(cache_size=5000, class_weight=None, coef0=0.0,
   decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
   max_iter=-1, nu=0.9, probability=False, random_state=None,
   shrinking=True, tol=0.001, verbose=False)
Number of un-Detected Attacks (FN) in Training Set is: FN =  11470  out of 125973
Number of False Attack Alarms (FP) in Training Set is: FP = 303  out of 125973
Accuracy on Trainin Set: Accuracy =  90.6543465663277 %

======================Results of KDDTest+ Set: ===================================

Number of un-Detected Attacks (FN) in KDDTest+ Set is: FN = 5916  out of 22543
Number of False Attack Alarms (FP) in KDDTest+ Set is: FP = 138  out of 22543
Accuracy on Trainin Set: Accuracy =  73.14465687796655 %

======================Results of KDDTest-21 Set: ===================================

Number of un-Detected Attacks (FN) in Test21 Set is: FN =  5872  out of 11850
Number of False Attack Alarms (FP) in Test21 Set is: FP = 139  out of 11850
Accuracy on Trainin Set: Accuracy =  49.27426160337552 %
>>> 
