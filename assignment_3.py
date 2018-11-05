import sklearn as skl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples,silhouette_score
from sklearn.decomposition import PCA, FastICA as ICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import FeatureAgglomeration
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import validation_curve, learning_curve, GridSearchCV

from scipy.stats import kurtosis
from numpy import linalg

import time

class Unsupe(object):
    def __init__(self, train_x, test_x):
        self.train = train_x
        self.test = test_x
        self.Clusters = None
        self.EM = None
        self.PCA = None
        self.ICA = None
        self.RAND_GUAS = None

    def kclusters(self, k, data = None, dim_reduct = None):
        if data is None:
            data = self.train

        else:
            data = pd.DataFrame(data)
        '''
        Initialize the KMeans instance
        Fit the training data to the KMeans instance
        Determine the score of the K clusters used
        '''
        clust = KMeans(k)
        clust.fit(data)
        clust_score = clust.score(data)
        print ('Cluster score for {} K is {}').format(k, clust_score)
        self.clusters = clust
        data['labels'] = clust.labels_

        # f, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4, sharex='col', sharey='row')
        #
        # for x in range(0, k):
        #     print x
        #     class_name = 'Class ' + str(x)
        #     # plt.scatter(data[x_value].loc[data['labels'] == x], data[y_value].loc[data['labels'] == x],
        #     #             c=np.random.rand(3, ), label=class_name)
        #
        #
        #     ax1.scatter(data.iloc[:,0].loc[data['labels'] == x], data.iloc[:,0].loc[data['labels'] == x],
        #                 c=np.random.rand(3, ), label=class_name)
        #     ax2.scatter(data.iloc[:,0].loc[data['labels'] == x], data.iloc[:,1].loc[data['labels'] == x],
        #                 c=np.random.rand(3, ), label=class_name)
        #     ax3.scatter(data.iloc[:,0].loc[data['labels'] == x], data.iloc[:,2].loc[data['labels'] == x],
        #                 c=np.random.rand(3, ), label=class_name)
        #     ax4.scatter(data.iloc[:,0].loc[data['labels'] == x], data.iloc[:,3].loc[data['labels'] == x],
        #                 c=np.random.rand(3, ), label=class_name)
        #
        # ax1.scatter(clust.cluster_centers_[:, 0], clust.cluster_centers_[:, 0], marker='+', s=169, linewidths=5, c='black')
        # ax2.scatter(clust.cluster_centers_[:, 0], clust.cluster_centers_[:, 1], marker='+', s=169, linewidths=5,
        #             c='black')
        # ax3.scatter(clust.cluster_centers_[:, 0], clust.cluster_centers_[:, 2], marker='+', s=169, linewidths=5,
        #             c='black')
        # ax4.scatter(clust.cluster_centers_[:, 0], clust.cluster_centers_[:, 3], marker='+', s=169, linewidths=5,
        #             c='black')
        # ax1.set(xlabel = data.iloc[:,0].name, ylabel = data.iloc[:,0].name)
        # ax2.set(xlabel = data.iloc[:,0].name, ylabel = data.iloc[:,1].name)
        # ax3.set(xlabel = data.iloc[:,0].name, ylabel = data.iloc[:,2].name)
        # ax4.set(xlabel = data.iloc[:,0].name, ylabel = data.iloc[:,3].name)
        #
        # if dim_reduct == None:
        #     f.suptitle('KMeans')
        # else:
        #     plt.title("{}: K vs Cluster SSE".format(dim_reduct))

        # plt.figure(2)
        # for x in range(0, k):
        #     plt.plot(clust.cluster_centers_[x], linestyle = 'x')
        #
        # data_min = np.min(data.drop(columns='labels'))
        # data_max = np.max(data.drop(columns = 'labels'))
        # data_mean = np.mean(data.drop(columns='labels'))
        #
        # # plt.scatter(data_min.index, data_min, marker='_', color = 'black')
        # # plt.scatter(data_max.index, data_max, marker='_', color = 'black')
        # plt.scatter(data_mean.index, data_mean, marker='x', color = 'black')
        # plt.xticks(rotation=90)
        # plt.show()
        self.Clusters = clust

    def find_kclusters(self, sweep_k, step, data = None, dim_reduct = None):
        if data is None:
            data = self.train
        else:
            data = pd.DataFrame(data)
        k_sweep_df = pd.DataFrame()
        ss = ShuffleSplit(n_splits=10, test_size=0.2)
        start = time.time()
        for x in range (2, sweep_k+2, step):
            print (x)
            data_length = len(data)
            sse_train = 0
            sse_cv = 0
            sil_train = 0
            sil_cv = 0
            # for train, cv in ss.split(data):
            clust_train = KMeans(x)
            clust_train.fit(data)
            sse_train += clust_train.inertia_
            sil_train += silhouette_score(data[:data_length/4], clust_train.labels_[:data_length/4])
            sil_train += silhouette_score(data[data_length/4:2*data_length/4], clust_train.labels_[data_length/4:2*data_length/4])
            sil_train += silhouette_score(data[2*data_length/4:3*data_length/4], clust_train.labels_[2*data_length/4:3*data_length/4])
            sil_train += silhouette_score(data[3*data_length/4:4*data_length/4], clust_train.labels_[3*data_length/4:4*data_length/4])




            # clust_cv = KMeans(x)
            # clust_cv.fit(data.iloc[cv])
            sil_train = sil_train/4
            # sse_cv += clust_cv.inertia_
            # sil_cv += silhouette_score(data.iloc[cv], clust_cv.labels_)

            # sse_cv = sse_cv/10
            # sil_cv = sil_cv/10
            k_sweep_df = k_sweep_df.append({"K": x, "SSE_Train": sse_train,
                                            "Sil_Train": sil_train}, ignore_index=True)

        print k_sweep_df
        print ("Finding KMean Clusters took: {} secs".format(str(time.time() - start)))

        fig, ax1 = plt.subplots()

        # color = 'tab:red'
        ax1.set_xlabel('K')
        ax1.set_ylabel('Silhouette Score', color='red')
        ax1.plot(k_sweep_df['K'],k_sweep_df['Sil_Train'], color='red', label = 'Train')
        # ax1.plot(k_sweep_df['K'],k_sweep_df['Sil_CV'], color='blue', label = 'Train')
        ax1.tick_params(axis='y', labelcolor='red')

        # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        # color = 'tab:blue'
        # ax2.set_ylabel('Silhouette Score', color=color)  # we already handled the x-label with ax1
        # ax2.plot(k_sweep_df['K'], k_sweep_df['Silhouette_Avg'], color=color)
        # ax2.tick_params(axis='y', labelcolor=color)


        plt.grid(True)
        if dim_reduct == None:
            plt.title("K vs Silhouette Score")
        else:
            plt.title("{}: K vs Silhouette Score".format(dim_reduct))
        plt.show()

    def gmm(self, k, data = None, dim_reduct = None):
        if data is None:
            data = self.train
        else:
            data = pd.DataFrame(data)

        guas = GaussianMixture(k)
        guas.fit(data)
        data['labels'] = guas.predict(data)

        # f, ((ax1, ax2 ,ax3, ax4)) = plt.subplots(1, 4, sharex='col', sharey='row')

        # for x in range(0, k):
        #     print x
        #     class_name = 'Class ' + str(x)
        #     # plt.scatter(data[x_value].loc[data['Guas_Labels'] == x],
        #     #             data[y_value].loc[data['Guas_Labels'] == x],
        #     #             c=np.random.rand(3, ), label=class_name)
        #     ax1.scatter(data.iloc[:,0].loc[data['labels'] == x], data.iloc[:,1].loc[data['labels'] == x],
        #                 c=np.random.rand(3, ), label=class_name)
        #     ax2.scatter(data.iloc[:,0].loc[data['labels'] == x], data.iloc[:,2].loc[data['labels'] == x],
        #                 c=np.random.rand(3, ), label=class_name)
        #     ax3.scatter(data.iloc[:,0].loc[data['labels'] == x], data.iloc[:,3].loc[data['labels'] == x],
        #                 c=np.random.rand(3, ), label=class_name)
        #     ax4.scatter(data.iloc[:,0].loc[data['labels'] == x], data.iloc[:,4].loc[data['labels'] == x],
        #                 c=np.random.rand(3, ), label=class_name)
        #
        # ax1.scatter(guas.means_[:, 0], guas.means_[:, 1], marker='+', s=169, linewidths=5, c='black')
        # ax2.scatter(guas.means_[:, 0], guas.means_[:, 2], marker='+', s=169, linewidths=5, c='black')
        # ax3.scatter(guas.means_[:, 0], guas.means_[:, 3], marker='+', s=169, linewidths=5, c='black')
        # ax4.scatter(guas.means_[:, 0], guas.means_[:, 4], marker='+', s=169, linewidths=5, c='black')
        #
        # ax1.set(xlabel = data.iloc[:,0].name, ylabel = data.iloc[:,1].name)
        # ax2.set(xlabel = data.iloc[:,0].name, ylabel = data.iloc[:,2].name)
        # ax3.set(xlabel = data.iloc[:,0].name, ylabel = data.iloc[:,3].name)
        # ax4.set(xlabel = data.iloc[:,0].name, ylabel = data.iloc[:,4].name)
        # # plt.xlabel(x_value)
        # # plt.ylabel(y_value)
        #
        # if dim_reduct == None:
        #     f.suptitle('GMM')
        # else:
        #     plt.title("GMM: {}".format(dim_reduct))
        # plt.show()
        self.EM = guas
        self.EM_labels = data['labels']

    def find_gmm(self, k, jump, data = None, dim_reduct = None):
        if data is None:
            data = self.train
        else:
            data = pd.DataFrame(data)

        guas_df = pd.DataFrame()
        ss = ShuffleSplit(n_splits=10, test_size=0.2)
        start = time.time()
        for x in range(1, k+1, jump):
            aic_train = 0
            aic_cv = 0

            bic_train = 0
            bic_cv = 0
            for train, cv in ss.split(data):
                guas = GaussianMixture(n_components= x, covariance_type='full', random_state=0)
                guas.fit(data.iloc[train])
                guas_aic = guas.aic(data.iloc[train])
                guas_bic = guas.bic(data.iloc[train])

                aic_train += guas_aic
                bic_train += guas_bic

                guas.fit(data.iloc[cv])
                guas_aic = guas.aic(data.iloc[cv])
                guas_bic = guas.bic(data.iloc[cv])

                aic_cv += guas_aic
                bic_cv += guas_bic

            aic_train = aic_train/10
            aic_cv = aic_cv/10

            bic_train = bic_train/10
            bic_cv = bic_cv/10
            guas_df = guas_df.append({"K": x, "Guas_Bic_Train": bic_train, "Guas_Aic_Train": aic_train,
                                      "Guas_Bic_CV": bic_cv, "Guas_Aic_CV": aic_cv}, ignore_index=True)


        print guas_df
        print ("Finding EM clusters took {} seconds".format(time.time() - start))
        plt.plot(guas_df['K'], guas_df['Guas_Bic_Train'], label = 'BIC_Train')
        plt.plot(guas_df['K'], guas_df['Guas_Aic_Train'], label = 'AIC_Train')
        plt.plot(guas_df['K'], guas_df['Guas_Bic_CV'], label = 'BIC_CV')
        plt.plot(guas_df['K'], guas_df['Guas_Aic_CV'], label = 'AIC_CV')
        plt.legend()
        plt.xlabel('K')
        plt.ylabel('AIC/BIC Score')
        if dim_reduct == None:
            plt.title("K vs AIC/BIC Score")
        else:
            plt.title("K vs AIC/BIC Score: {}".format(dim_reduct))
        plt.show()

    def find_pca(self):
        pct_ranges = [30]
        pca_df = pd.DataFrame()
        ss = ShuffleSplit(n_splits=10, test_size=0.2)
        data = self.train
        for x in range (0, len(pct_ranges)):
            pca_variance_train = 0
            pca_variance_cv = 0
            for train, cv in ss.split(data):
                pca_features = PCA(pct_ranges[x], svd_solver='full')
                pca_features.fit(data.iloc[train])
                pca_variance_train = np.cumsum(pca_features.explained_variance_ratio_)

                pca_features.fit(data.iloc[cv])
                pca_variance_cv = np.cumsum(pca_features.explained_variance_ratio_)

            pca_df = pca_df.append({"pca_pct": pct_ranges[x], "pca_train": pca_variance_train, "pca_cv": pca_variance_cv}, ignore_index=True)

        plt.plot(pca_variance_train, label = 'PCA_Train')
        plt.plot(pca_variance_cv, label = 'PCA_CV')
        plt.title('PCA: N Components vs Variance')
        plt.xlabel('PCA Components')
        plt.ylabel('Variance')
        plt.legend()
        plt.grid()
        plt.show()

    def pca(self, pct, data = None):
        if data is None:
            data = self.train

        # Initializing the PCA instance with a percentage is telling the algorithm how much variance we wish to keep within the
        # dataset. Dropping below 80% we would be losing a lot of data
        pca_features = PCA(pct)
        pca_features.fit(data)
        pca_features.score(data)
        print (pca_features.explained_variance_)
        self.pca_train_data = pca_features.transform(data)
        n_samples = data.shape[0]
        X_centered = data - np.mean(data, axis=0)
        cov_matrix = np.dot(X_centered.T, X_centered) / n_samples
        eigenvalues = pca_features.explained_variance_
        for eigenvalue, eigenvector in zip(eigenvalues, pca_features.components_):
            print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
            print(eigenvalue)

        self.PCA = pca_features

        pca_test = PCA(pct)
        pca_test.fit(self.test)
        self.pca_test_data = pca_test.transform(self.test)

    '''
    Need an ICA, Randomized Projections and another feature selection 
    '''
    def find_ica(self, n_comp, data = None):
        if data is None:
            data = self.train
        ica_kutosis = pd.DataFrame()
        ss = ShuffleSplit(n_splits=10, test_size=0.2)
        for x in range(2, n_comp+1):
            ica_kutosis_train = 0
            ica_kutosis_cv = 0
            for train, cv in ss.split(self.train):
                ica_features = ICA(n_components= x, algorithm='parallel',max_iter=500)
                ica_features.fit(data.iloc[train])
                ica_train_data = ica_features.transform(data.iloc[train])
                ica_kutosis_train += np.sum(np.abs(kurtosis(ica_train_data)))

                ica_features.fit(data.iloc[cv])
                ica_cv_data = ica_features.transform(data.iloc[cv])
                ica_kutosis_cv += np.sum(np.abs(kurtosis(ica_cv_data)))

            ica_kutosis_train = ica_kutosis_train/10
            ica_kutosis_cv = ica_kutosis_cv/10
            ica_kutosis = ica_kutosis.append({"K": x, "Kurtosis_Train": ica_kutosis_train, "Kurtosis_CV": ica_kutosis_cv}, ignore_index=True)

        print ica_kutosis
        plt.plot(ica_kutosis["K"], ica_kutosis["Kurtosis_Train"], color = 'red', label = "Train")
        plt.plot(ica_kutosis["K"], ica_kutosis["Kurtosis_CV"], color = 'blue', label = "CV")
        plt.xlabel("N Components")
        plt.ylabel("Absolute Sum Kurtosis")
        plt.title("ICA: N Components vs Kurtosis")
        plt.grid()
        plt.legend()
        plt.show()

    def ica(self, n_comp, data = None):
        if data is None:
            data = self.train

        # Initializing the PCA instance with a percentage is telling the algorithm how much variance we wish to keep within the
        # dataset. Dropping below 80% we would be losing a lot of data
        ica_features = ICA(n_comp)
        ica_features.fit(data)
        self.ica_train_data = ica_features.transform(data)
        self.ICA = ica_features

        ica_test = ICA(n_comp)
        ica_test.fit(self.test)
        self.ica_test_data = ica_test.transform(self.test)


    def find_rand_guas(self, n_comp, data = None):
        if data is None:
            data = self.train

        error = pd.DataFrame()
        ss = ShuffleSplit(n_splits=10, test_size=0.2)
        for x in range(2, n_comp):
            total_loss_train = 0
            total_loss_cv = 0
            for train, cv in ss.split(data):
                init_comp = data.iloc[train].shape[1]
                rand_guas = GaussianRandomProjection(n_components=x)
                rand_guas.fit(data.iloc[train])
                rand_guas_train = rand_guas.transform(data.iloc[train])

                inv_guas = GaussianRandomProjection(n_components=init_comp)
                inv_guas.fit(rand_guas_train)
                inv_data = inv_guas.transform(rand_guas_train)
                total_loss_train += linalg.norm(data.iloc[train] - inv_data, None)

                rand_guas = GaussianRandomProjection(n_components=x)
                rand_guas.fit(data.iloc[cv])
                rand_guas_cv = rand_guas.transform(data.iloc[cv])

                inv_guas = GaussianRandomProjection(n_components=init_comp)
                inv_guas.fit(rand_guas_cv)
                inv_data = inv_guas.transform(rand_guas_cv)
                total_loss_cv += linalg.norm(data.iloc[cv] - inv_data, None)

            total_loss_train = total_loss_train/10
            total_loss_cv = total_loss_cv/10
            error = error.append({"K": x, "Train_Loss": total_loss_train, "CV_Loss": total_loss_cv}, ignore_index=True)

        plt.plot(error["K"], error["Train_Loss"], color = 'red', label = 'Train')
        plt.plot(error["K"], error["CV_Loss"], color = 'blue', label = 'CV')
        plt.title("Rand Gaussian: N Components vs. Reconstruction Error")
        plt.legend()
        plt.grid()
        plt.show()

    def rand_guas(self, n_comp, data = None):
        if data is None:
            data = self.train
        else:
            data = pd.DataFrame(data)

        rand_guas = GaussianRandomProjection(n_components=n_comp)
        rand_guas.fit(data)
        self.rand_guas_train_data = rand_guas.transform(data)
        self.RAND_GUAS = rand_guas

        rand_test = GaussianRandomProjection(n_components=n_comp)
        rand_test.fit(self.test)
        self.rand_guas_test_data = rand_test.transform(self.test)

    def find_rand_forrest(self, train_data, test_data = None):
        agg_var = pd.DataFrame()

        rand_forr = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        rand_forr.fit(self.train, train_data)
        print (rand_forr.feature_importances_)

        self.rand_forr_features = np.argsort(rand_forr.feature_importances_)[-6:]

        if test_data is not None:
            rand_forr = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
            rand_forr.fit(self.test, test_data)
            self.rand_forr_features_test = np.argsort(rand_forr.feature_importances_)[-6:]

        plt.bar(self.train.columns, rand_forr.feature_importances_)
        plt.xlabel("Feature")
        plt.xticks(rotation = 90)
        plt.ylabel("Feature Importance")
        plt.title("Random Forrest Feature Importance")
        plt.show()

    def adult_nn(self, x_train, y_train):
        train_size, learn_train_score, learn_test_score = learning_curve(MLPClassifier(activation='identity',
                                                                                       solver='adam',
                                                                                       early_stopping=False,
                                                                                       hidden_layer_sizes=(5, 10),
                                                                                       learning_rate_init=0.001,
                                                                                       batch_size=300,
                                                                                       verbose=3),
                                                                         x_train,
                                                                         y_train,
                                                                         train_sizes=np.linspace(0.1, 1, 10),
                                                                         cv=5,
                                                                         shuffle=True)

        plt.plot(train_size, np.ndarray.mean(learn_train_score, axis=1), 'r^-', label='Training Score')
        plt.plot(train_size, np.ndarray.mean(learn_test_score, axis=1), label='CV Score')
        plt.title("adult: Learning Curve - Neural Net")
        plt.legend(loc='best')
        plt.show()

    def posture_nn(self, x_train, y_train, x_test, y_test, data_type = None):
        train_size, learn_train_score, learn_test_score = learning_curve(MLPClassifier(activation='identity',
                                                                                       solver='adam',
                                                                                       max_iter=200,
                                                                                       alpha=0.0001,
                                                                                       hidden_layer_sizes=(100, 50)),
                                                                         x_train,
                                                                         y_train,
                                                                         cv=5,
                                                                         shuffle=True)

        print train_size, learn_train_score, learn_test_score

        plt.plot(train_size, np.ndarray.mean(learn_train_score, axis=1), 'r^-', label='Training Score')
        plt.plot(train_size, np.ndarray.mean(learn_test_score, axis=1), label='CV Score')
        if data_type == None:
            plt.title("Posture: Learning Curve - Neural Net")
        else:
            plt.title("Posture: Learning Curve - Neural Net ({})".format(data_type))
        plt.legend(loc='best')
        plt.grid()
        plt.show()

        post_NN = MLPClassifier(activation='identity',
                                solver='adam',
                                max_iter=200,
                                alpha=0.0001,
                                hidden_layer_sizes=(100, 50))
        start = time.time()
        post_NN.fit(x_train, y_train)
        timed = time.time() - start
        print ("Post NN time %s", timed)
        score = post_NN.score(x_test, y_test)
        print ("Post NN score for {}: {}".format(data_type, str(score)))


def adult_data_setup():
    adult_set = pd.read_csv('adult_data.csv',
                            header = None,
                            usecols = [0, 1, 3, 5, 6, 8, 9, 12, 14],
                            names = ['age', 'work-class', 'education',
                                     'martial-status', 'occupation', 'race',
                                     'sex', 'hrs_per_week', 'more_than_50k'])
    # adult_set = adult_set.loc[adult_set['work-class'] != ' ?']

    adult_set.replace(' ?', 'Missing', inplace = True)

    # Separate majority and minority classes
    df_majority = adult_set[adult_set.more_than_50k == ' <=50K']
    df_minority = adult_set[adult_set.more_than_50k == ' >50K']

    # Upsample minority class
    df_majority_upsampled = resample(df_majority,
                                     replace=False,  # sample with replacement
                                     n_samples=7650,  # to match majority class
                                     random_state=123)  # reproducible results

    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_minority, df_majority_upsampled])

    df_upsampled.more_than_50k.value_counts()

    adult_set = df_upsampled

    adult_set = pd.get_dummies(adult_set)
    adult_set.dropna(inplace=True)

    scale = MinMaxScaler()
    scale.fit(adult_set[['age', 'hrs_per_week']])
    adult_set[['age', 'hrs_per_week']] = scale.transform(adult_set[['age', 'hrs_per_week']])

    adult_set.drop(columns=['more_than_50k_ >50K', 'sex_ Female'], inplace=True)
    adult_set_x = adult_set.drop(columns = 'more_than_50k_ <=50K')
    adult_set_y = adult_set['more_than_50k_ <=50K']
    x_train, x_test, y_train, y_test = train_test_split(adult_set_x, adult_set_y,
                                                        train_size = 0.8, shuffle = True, stratify = adult_set_y)


    return x_train, y_train, x_test, y_test

def posture_data_setup():
    data = pd.read_csv("Postures.csv", error_bad_lines=False, header=0)
    # data = data[[u'Class', u'User', u'X0', u'Y0', u'Z0', u'X1', u'Y1', u'Z1', u'X2',
    #              u'Y2', u'Z2']]
    data = data.replace('?', float(-999))

    data = data.drop(data.index[0])
    data_filter = data.loc[data['User'] <= 5]
    data_filter = data.loc[data['Class'] <= 3]

    for i in range(0, 12):
        x = str('X' + str(i));
        y = str('Y' + str(i));
        z = str('Z' + str(i));
        data_filter[x] = pd.to_numeric(data_filter[x])
        data_filter[y] = pd.to_numeric(data_filter[y])

    pos_data = data_filter

    pos_data_keep = pos_data[['User', 'Class']]

    scale =  StandardScaler()
    scale.fit(pos_data[pos_data.columns])
    pos_data[pos_data.columns] = scale.transform(pos_data[pos_data.columns])

    pos_data['User'] = pos_data_keep['User']
    pos_data['Class'] = pos_data_keep['Class']

    pos_train = pos_data.loc[pos_data['User'] != 5]
    pos_test = pos_data.loc[pos_data['User'] == 5]
    #
    pos_train = pos_train.sample(frac=1).reset_index(drop=True)
    pos_test = pos_test.sample(frac=1).reset_index(drop=True)


    pos_train_x = pos_train.drop(columns=['Class'])
    pos_train_y = pos_train['Class']

    pos_test_x = pos_test.drop(columns = ['Class'])
    pos_test_y = pos_test['Class']

    return pos_train_x, pos_train_y, pos_test_x, pos_test_y

if __name__ == "__main__":

    train_x, train_y, test_x, test_y = posture_data_setup()
    train_x = train_x.drop(columns = 'User')
    test_x = test_x.drop(columns='User')
    post = Unsupe(train_x, test_x)

    # 6 clusters
    post.find_kclusters(10, 2)
    post.find_gmm(20)
    post.find_pca()
    post.kclusters(4)
    post.gmm(6)

    # 6 PCA Clusters
    post.find_pca()

    # 7 ICA Clusters
    post.find_ica(20)
    post.pca(6)
    post.ica(3)
    post.find_rand_guas(10)
    post.rand_guas(3)
    post.find_rand_forrest(train_y, test_data= test_y)
    random_forr_data = post.train[post.train.columns[post.rand_forr_features]]
    random_forr_test = post.test[post.test.columns[post.rand_forr_features_test]]

    pca_train = pd.DataFrame(data=post.pca_train_data)
    pca_test = pd.DataFrame(data = post.pca_test_data)
    ica_train = pd.DataFrame(data = post.ica_train_data)
    ica_test = pd.DataFrame(data = post.ica_test_data)
    rand_guas_train = pd.DataFrame(data = post.rand_guas_train_data)
    rand_guas_test = pd.DataFrame(data = post.rand_guas_test_data)
    random_forrest_train = pd.DataFrame(data = random_forr_data)
    random_forrest_test = pd.DataFrame(data = random_forr_test)

    post.find_gmm(20, 2, data= post.pca_train_data, dim_reduct= "PCA")
    post.find_gmm(20, 2, data= post.ica_train_data, dim_reduct= "ICA")
    post.find_gmm(20, 2, data= post.rand_guas_train_data, dim_reduct= "RAND GAUS")
    post.find_gmm(20, 2, data = random_forr_data, dim_reduct="Random Forrest/GMM")

    post.gmm(6, data= post.pca_train_data, dim_reduct= "PCA")
    pca_gmm = post.EM_labels
    post.gmm(6, data= post.ica_train_data, dim_reduct= "ICA")
    ica_gmm = post.EM_labels
    post.gmm(6, data= post.rand_guas_train_data, dim_reduct= "RAND GAUS")
    rand_guas_gmm = post.EM_labels
    post.gmm(6, data= random_forr_data, dim_reduct= "Random Forrest")
    forrest_gmm = post.EM_labels

    pca_train['labels'] = pca_gmm
    ica_train['labels'] = ica_gmm
    rand_guas_train['labels'] = rand_guas_gmm
    random_forrest_train['labels'] = forrest_gmm
    #
    post.posture_nn(pca_train, train_y, "PCA/GMM")
    post.posture_nn(ica_train, train_y, "ICA/GMM")
    post.posture_nn(rand_guas_train, train_y, "RAND GUAS/GMM")
    post.posture_nn(random_forrest_train, train_y, "Random Forrest/GMM")

    post.find_kclusters(10, 2, data= post.pca_train_data, dim_reduct="PCA")
    post.find_kclusters(10, 2, data = post.ica_train_data, dim_reduct="ICA")
    post.find_kclusters(10, 2, data = post.rand_guas_train_data, dim_reduct="RAND GUAS")
    post.find_kclusters(10, 2, data = random_forr_data, dim_reduct="Random Forrest/KMeans")
    post.kclusters(4, post.pca_train_data, dim_reduct= "PCA")
    pca_clust = post.Clusters.labels_
    post.kclusters(2, post.ica_train_data, dim_reduct= "ICA")
    ica_clust = post.Clusters.labels_
    post.kclusters(6, post.rand_guas_train_data, dim_reduct= "RAND GAUS")
    rand_guas_clust = post.Clusters.labels_
    post.kclusters(4, random_forr_data, dim_reduct= "Random Forrest")
    forrest_clust = post.Clusters.labels_

    post.kclusters(4, post.pca_test_data, dim_reduct= "PCA")
    pca_clust_test = post.Clusters.labels_
    post.kclusters(2, post.ica_test_data, dim_reduct= "ICA")
    ica_clust_test = post.Clusters.labels_
    post.kclusters(6, post.rand_guas_test_data, dim_reduct= "RAND GAUS")
    rand_guas_clust_test = post.Clusters.labels_
    post.kclusters(4, random_forrest_test, dim_reduct= "Random Forrest")
    forrest_clust_test = post.Clusters.labels_
    post.gmm(10, post.pca_train_data, dim_reduct="PCA")
    post.gmm(3, random_forr_data, dim_reduct="Random Forrest")

    # #Posture PCA NN
    post.posture_nn(pca_train, train_y, pca_test, test_y, "PCA")
    pca_train['labels'] = pca_clust
    pca_test['labels'] = pca_clust_test
    # #Posture PCA/KMeans NN
    post.posture_nn(pca_train, train_y, pca_test, test_y, "PCA/KMeans")
    #
    #Posture ICA NN
    post.posture_nn(ica_train, train_y, ica_test, test_y, "ICA")
    ica_train['lables'] = ica_clust
    ica_test['labels'] = ica_clust_test
    #Posture ICA/KMeans NN
    post.posture_nn(ica_train, train_y, ica_test, test_y, "ICA/KMeans")
    #
    #Posture Rand Gaussian NN
    post.posture_nn(rand_guas_train, train_y, rand_guas_test, test_y, "RAND GAUS")
    #Posture Rand Gaussian/Kmeans NN
    rand_guas_train['lables'] = rand_guas_clust
    rand_guas_test['labels'] = rand_guas_clust_test
    post.posture_nn(rand_guas_train, train_y, rand_guas_test, test_y, "RAND GAUS/KMeans")

    #Posture Rand Gaussian NN
    post.posture_nn(random_forrest_train, train_y, random_forrest_test, test_y, "Random Forrest")
    #Posture Rand Gaussian/Kmeans NN
    random_forrest_train['lables'] = forrest_clust
    random_forrest_test['labels'] = forrest_clust_test
    post.posture_nn(random_forrest_train, train_y, random_forrest_test, test_y, "Random Forrest/KMeans")
    '''
    Setup the adult dataset 
    Initialize the Unsupe class 
    Find and plot the ranges or clusters 
    
    Use a confusion maxtix for some of the methods
    '''
    train_x, train_y, test_x, test_y = adult_data_setup()
    adult = Unsupe(train_x, test_x)

    #6  KClust
    adult.find_kclusters(200, 50)
    adult.find_gmm(50, 10)

    #PCA 16 components
    adult.find_pca()

    #ICA 47 components
    adult.find_ica(50)

    #3 components from reconstruction
    adult.find_rand_guas(20)

    adult.find_rand_forrest(train_y)
    random_forr_data = adult.train[adult.train.columns[adult.rand_forr_features]]

    adult.kclusters(6)
    adult.gmm(6)
    adult.pca(16)
    adult.ica(47)
    adult.rand_guas(15)
    adult.find_kclusters(50, 5, data = adult.pca_train_data, dim_reduct= "PCA/KMeans")
    adult.find_kclusters(50, 5, data = adult.ica_train_data, dim_reduct= "ICA/KMeans")
    adult.find_kclusters(50, 5, data=random_forr_data, dim_reduct="Random Forrest/KMeans")
    adult.find_kclusters(50, 5, data = adult.rand_guas_train_data, dim_reduct= "RP/KMeans")

    #
    # adult.kclusters(20, data = adult.pca_train_data, dim_reduct= "PCA")
    # pca_clust = adult.Clusters.labels_
    # adult.kclusters(30, data = adult.ica_train_data, dim_reduct= "ICA")
    # ica_clust = adult.Clusters.labels_
    # adult.kclusters(20, data = adult.rand_guas_train_data, dim_reduct="RP")
    # adult.kclusters(6, data = random_forr_data, dim_reduct="Random Forrest")

    #
    adult.find_gmm(50, 5, data = adult.pca_train_data, dim_reduct= "PCA/GMM")
    adult.find_gmm(50, 5, data = adult.ica_train_data, dim_reduct= "ICA/GMM")
    adult.find_gmm(50, 5, data = adult.rand_guas_train_data, dim_reduct= "RP/GMM")
    adult.find_gmm(20, 2, data= random_forr_data, dim_reduct="Random Forrest/GMM")

    adult.gmm(20, data = adult.pca_train_data, dim_reduct= "PCA/GMM")
    adult.gmm(7, data = adult.ica_train_data, dim_reduct= "ICA/GMM")
    adult.gmm(20, data=adult.rand_guas_train_data, dim_reduct="RP/GMM")
    adult.gmm(10, data=adult.rand_guas_train_data, dim_reduct="Random Forrest/GMM")
    #
    # adult.adult_nn(adult.pca_train_data, train_y)
    # adult.adult_nn(adult.ica_train_data, train_y)
    #
    # pca_train = pd.DataFrame(data = adult.pca_train_data)
    # pca_train['labels'] = pca_clust
    # adult.adult_nn(pca_train, train_y)
    #
    # ica_train = pd.DataFrame(data = adult.ica_train_data)
    # ica_train['lables'] = ica_clust
    # adult.adult_nn(ica_train, train_y)




