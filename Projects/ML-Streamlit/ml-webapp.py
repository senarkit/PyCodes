import streamlit as st
import os
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# -------- Changing Current Directory ----------#
print("Current Working Directory : ", os.getcwd())
if str(os.getcwd()).split("/")[-1] != "ML-Streamlit":
    os.chdir("../Projects/ML-Streamlit/")
    print("Changed to : ", os.getcwd())

# Title
st.title("Streamlit Example")
st.write(
    """
## Explore different classifier
Which one is the best ?
"""
)

# ----------- Read data -----------------------#
dataset_nm = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine"))
classifier_nm = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))


def get_data(dataset_nm):
    if dataset_nm == "IRIS":
        data = datasets.load_iris()
    elif dataset_nm == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()

    X = data.data
    y = data.target
    return X, y


data, target = get_data(dataset_nm)
st.write("Shape of Dataset", data.shape)
st.write("Number of Classes", len(np.unique(target)))

# ---------- Add Classifier Parameters -----------#


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params


params = add_parameter_ui(classifier_nm)
# st.write("Parameters : ", params.keys(), params.values())


def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(
            max_depth=params["max_depth"], n_estimators=params["n_estimators"], random_state=1234
        )
    return clf


clf = get_classifier(classifier_nm, params)

# ---------- Classification -----------#
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2, random_state=3)

clf.fit(data_train, target_train)
target_pred = clf.predict(data_test)

acc = accuracy_score(target_test, target_pred)
st.write(f"Classifier = {classifier_nm}")
st.write(f"Accuracy = {round(acc,3)}")


# PLOT
pca = PCA(2)
data_projected = pca.fit_transform(data)

data1 = data_projected[:, 0]
data2 = data_projected[:, 1]

fig = plt.figure()
plt.scatter(data1, data2, c=target, alpha=0.8, cmap="viridis")
plt.xlabel("Pricinple Component 1")
plt.ylabel("Principle Component 2")
plt.colorbar()

st.pyplot(fig)
