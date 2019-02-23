from data import *
from sklearn import preprocessing, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

def trainmodel():
    X_train, X_test, y_train, y_test = train_test_gen('data.csv')
    # lr_clf = LogisticRegression(
    #     random_state = 200,
    #     max_iter = 1000,
    #     verbose = 1,
    #     n_jobs = -1,
    #     solver = 'newton-cg'
    # )
    # lr_clf.fit(X_train, y_train)
    # predicted = lr_clf.predict_proba(X_test)

    # knn_clf = KNeighborsClassifier(
    #     n_neighbors = 5,
    #     n_jobs = -1,
    #     leaf_size = 100
    # )
    # knn_clf.fit(X_train, y_train)
    # predicted = knn_clf.predict_proba(X_test)

    # svc_clf = svm.SVC(
    #     kernel = 'linear',
    #     verbose = True,
    #     random_state = True
    # )
    # svc_clf.fit(X_train, y_train)
    # pred_labels = svc_clf.predict(X_test)

    clf = svm.SVC(
        kernel = 'linear', 
        probability=True, 
        C = 10, 
        gamma = 0.1
    )
    clf.fit(X_train, y_train)
    accuracy = accuracy_score(clf.predict(X_test), y_test)

    # C_grid = [0.001, 0.01, 0.1, 1, 10]
    # gamma_grid = [0.001, 0.01, 0.1, 1, 10]
    # param_grid = {'C': C_grid, 'gamma' : gamma_grid}

    # grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv = 3, scoring = "accuracy")
    # grid.fit(X_train, y_train)

    # Find the best model
    # print(grid.best_score_)
    # print(grid.best_params_)
    # print(grid.best_estimator_)
    # pred_labels = predicted.argmax(axis = 1)
    # print(("Accuracy score")+str(accuracy_score(y_test, pred_labels)))
    return accuracy