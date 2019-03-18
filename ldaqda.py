
import numpy as np
import matplotlib.pyplot as plt
import util

def discrimAnalysis(x, y):
    """
    Estimate the parameters in LDA/QDA and visualize the LDA/QDA models
    
    Inputs
    ------
    x: a N-by-2 2D array contains the height/weight data of the N samples
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of five elments: mu_male,mu_female,cov,cov_male,cov_female
    in which mu_male, mu_female are mean vectors (as 1D arrays)
             cov, cov_male, cov_female are covariance matrices (as 2D arrays)
    Besides producing the five outputs, you need also to plot 1 figure for LDA 
    and 1 figure for QDA in this function         
    """
    ### TODO: Write your code here
    N = x.shape[0]
    #men and women
    men = [x[m] for m in range(N) if y[m] == 1]
    women = [x[w] for w in range(N) if y[w] == 2]
    men = np.array(men)
    women = np.array(women)

    #average of males and females, along with overall average
    mu_male = np.average(men, axis = 0).reshape(1,2)
    mu_female = np.average(women, axis=0).reshape(1,2)
    fullmean = np.average(x, axis = 0).reshape(1,2)

    #our covariance matrices
    cov = x.T@x/N - fullmean.T@fullmean
    cov_male = men.T@men/men.shape[0] - mu_male.T@mu_male
    cov_female = women.T@women/women.shape[0] - mu_female.T@mu_female

    #meshgrids for the height and weight values
    heights = np.linspace(50,80,100)
    weights = np.linspace(80,280,100)
    heightgrid, weightgrid = np.meshgrid(heights, weights)

    #plot the data
    cdict = {1: 'firebrick', 2: 'navy'}
    sdict = {1: 'Male', 2: 'Female'}
    #plt.scatter(x[:,0], x[:,1], c = y, s = 50, cmap = 'RdBu', label = y)
    fig, ax = plt.subplots()
    for g in np.unique(y):
        ix = np.where(y == g)
        ax.scatter(x[ix,0], x[ix,1], c=cdict[g], label=sdict[g], s=25, alpha = 0.7)
    ax.legend()

    #ravelled versions of the meshgrids
    h = heightgrid.ravel()
    w =  weightgrid.ravel()
    maledensity = np.zeros(h.shape)
    femaledensity = np.zeros(h.shape)
    #where we'll store the assigned sex
    sex = np.zeros(h.shape)

    for i in range(h.shape[0]):
        sample = np.array([[h[i]],[w[i]]])

        #calculate the LDA values for men and women, then assign sex based on them
        maledensity[i], femaledensity[i] = LinearDiscriminantAnalysis(sample,mu_male.T, mu_female.T, cov)
        if maledensity[i] - femaledensity[i] > 0:
            sex[i] = 1
        else:
            sex[i] = 2

    #plot out contour maps and decision boundary along with colors indicating the classification region
    maledensity = maledensity.reshape(heightgrid.shape)
    femaledensity = femaledensity.reshape(heightgrid.shape)
    sex = sex.reshape(heightgrid.shape)

    boundary = maledensity-femaledensity
    cont = plt.contour(heightgrid, weightgrid, boundary, [0])

    plt.pcolormesh(heightgrid, weightgrid, sex, cmap = 'RdBu', alpha = 0.1)
    plt.clabel(cont, fontsize = 8, fmt = r'LDA Boundary')

    #Plot LDA contour maps for Male and Female
    heightweight = np.concatenate((h.reshape(-1,1),w.reshape(-1,1)), axis = 1)
    LDAResM = util.density_Gaussian(mu_male, cov, heightweight)
    LDAResM = LDAResM.reshape(heightgrid.shape)
    ctm = plt.contour(heightgrid,weightgrid,LDAResM, colors = 'firebrick')
    #plt.clabel(ctm, fontsize = 8, fmt = r'LDA Male Contours')

    LDAResF = util.density_Gaussian(mu_female, cov, heightweight)
    LDAResF = LDAResF.reshape(heightgrid.shape)
    ctm = plt.contour(heightgrid,weightgrid,LDAResF, colors = 'navy')
    #plt.clabel(ctm, fontsize = 8, fmt = r'LDA Female Contours')

    plt.ylim(80,280)
    plt.title('LDA')
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.savefig('LDABoundary.png')
    plt.show()
    plt.clf()
    plt.close()

    #Same as before but with QDA
    fig, ax = plt.subplots()
    for g in np.unique(y):
        ix = np.where(y == g)
        ax.scatter(x[ix,0], x[ix,1], c=cdict[g], label=sdict[g], s=25, alpha = 0.7)
    ax.legend()

    maledensity = np.zeros(h.shape)
    femaledensity = np.zeros(h.shape)
    sex = np.zeros(h.shape)
    for i in range(h.shape[0]):
        sample = np.array([[h[i]],[w[i]]])
        maledensity[i], femaledensity[i] = QuadraticDiscriminantAnalysis(sample,mu_male.T, mu_female.T, cov_male, cov_female)
        if maledensity[i] - femaledensity[i] > 0:
            sex[i] = 1
        else:
            sex[i] = 2
    maledensity = maledensity.reshape(heightgrid.shape)
    femaledensity = femaledensity.reshape(heightgrid.shape)
    sex = sex.reshape(heightgrid.shape)

    boundary = maledensity-femaledensity
    cont = plt.contour(heightgrid, weightgrid, boundary, [0])

    QDAResM = util.density_Gaussian(mu_male, cov_male, heightweight)
    QDAResM = QDAResM.reshape(heightgrid.shape)
    ctm = plt.contour(heightgrid,weightgrid,QDAResM, colors = 'firebrick')
    #plt.clabel(ctm, fontsize = 8, fmt = r'LDA Male Contours')

    QDAResF = util.density_Gaussian(mu_female, cov_female, heightweight)
    QDAResF = QDAResF.reshape(heightgrid.shape)
    ctm = plt.contour(heightgrid,weightgrid,QDAResF, colors = 'navy')
    #plt.clabel(ctm, fontsize = 8, fmt = r'LDA Female Contours')

    plt.pcolormesh(heightgrid, weightgrid, sex, cmap = 'RdBu', alpha = 0.1)
    plt.clabel(cont, fontsize = 8, fmt = r'QDA Boundary')
    plt.ylim(80,280)
    plt.title('QDA')
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.savefig('QDA Boundary')
    plt.show()

    return (mu_male,mu_female,cov,cov_male,cov_female)
    
def LinearDiscriminantAnalysis(sample,mu_male,mu_female,cov):
    invcov = np.linalg.inv(cov)
    BetaM = invcov@mu_male
    GammaM = -0.5*mu_male.T@invcov@mu_male + np.log(0.5)

    BetaF = invcov@mu_female
    GammaF = -0.5*mu_female.T@invcov@mu_female + np.log(0.5)
    ymale = BetaM.T@sample + GammaM
    yfemale = BetaF.T@sample + GammaF

    return ymale, yfemale
def QuadraticDiscriminantAnalysis(sample, mu_male, mu_female, cov_male, cov_female):
    csampleM = sample - mu_male
    csampleF = sample - mu_female
    invcovM = np.linalg.inv(cov_male)
    invcovF = np.linalg.inv(cov_female)

    constantM = -0.5*np.log(np.linalg.det(cov_male))
    constantF = -0.5*np.log(np.linalg.det(cov_female))

    ymale = constantM - 0.5*csampleM.T@invcovM@csampleM
    yfemale = constantF - 0.5*csampleF.T@invcovF@csampleF

    return ymale, yfemale
def misRate(mu_male,mu_female,cov,cov_male,cov_female,x,y):
    """
    Use LDA/QDA on the testing set and compute the misclassification rate
    
    Inputs
    ------
    mu_male,mu_female,cov,cov_male,mu_female: parameters from discrimAnalysis
    
    x: a N-by-2 2D array contains the height/weight data of the N samples  
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of two elements: (mis rate in LDA, mis rate in QDA )
    """
    ### TODO: Write your code here
    LDA = np.zeros(y.shape)
    QDA = np.zeros(y.shape)
    for i in range(x.shape[0]):
        #sample is each data point, classify them based on LDA and QDA
        sample = x[i,:].reshape(2,1)
        male, female = LinearDiscriminantAnalysis(sample,mu_male.T, mu_female.T,cov)
        if male > female:
            LDA[i] = 1
        else:
            LDA[i] = 2
        male, female = QuadraticDiscriminantAnalysis(sample, mu_male.T, mu_female.T, cov_male, cov_female)
        if male > female:
            QDA[i] = 1
        else:
            QDA[i] = 2


    mis_lda =  (LDA != y).sum()/y.shape[0]
    mis_qda =  (QDA != y).sum()/y.shape[0]
    return (mis_lda, mis_qda)


if __name__ == '__main__':
    
    # load training data and testing data
    x_train, y_train = util.get_data_in_file('trainHeightWeight.txt')
    x_test, y_test = util.get_data_in_file('testHeightWeight.txt')
    
    # parameter estimation and visualization in LDA/QDA
    mu_male,mu_female,cov,cov_male,cov_female = discrimAnalysis(x_train,y_train)
    
    # misclassification rate computation
    mis_LDA,mis_QDA = misRate(mu_male,mu_female,cov,cov_male,cov_female,x_test,y_test)
    print(mis_LDA, mis_QDA)