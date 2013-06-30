close all
clear all
clc

% GLM = general linear model
% CCA = canonical correlation analysis
% CBA = classification based analysis

% Number of timepoints in test data
N = 200;

% Add activity in a block form,
% two of the subjects have activity A > B,
% two of the subjects have activity B > A,

% Create paradigm
paradigm = zeros(N,1);
NN = 0;
while NN < N
    paradigm((NN+1):(NN+10),1) =   1;  % Activity
    paradigm((NN+11):(NN+20),1) =  0;  % Rest
    NN = NN + 20;
end


% Convolve paradigm with hemodynamic response function
hrf = spm_hrf(2);
dhrf = diff(hrf);

activity = conv(paradigm,hrf);
activity = activity(1:N);

% Setup temporal regressors for GLM and CCA

% Paradigm convolved with hrf
GLM1 = conv(paradigm,hrf);
GLM1 = GLM1(1:N);

% Paradigm convolved with temporal derivative of hrf,
% to be able to adjust for BOLD delay
GLM2 = conv(paradigm,dhrf);
GLM2 = GLM2(1:N);

X_GLM = zeros(N,2);
X_GLM(:,1) = GLM1 - mean(GLM1);
X_GLM(:,2) = GLM2 - mean(GLM2);

% Normalize regressors
X_GLM(:,1) = X_GLM(:,1)/norm(X_GLM(:,1));
X_GLM(:,2) = X_GLM(:,2)/norm(X_GLM(:,2));

% Orthogonalize regressors
X_GLM(:,2) = X_GLM(:,2) - (X_GLM(:,1)'*X_GLM(:,2))*X_GLM(:,1);

% Contrast vector
c = [1 ; 0];
ctxtx_GLM = c'*inv(X_GLM'*X_GLM)*c;


mean_beta_GLM = 0;
mean_ttest_GLM = 0;
mean_beta_CCA = 0;
mean_classification_performance = 0;
mean_canon_corr = 0;

classification_performance1 = 0;
classification_performance2 = 0;
classification_performance3 = 0;
classification_performance4 = 0;

mean_gamma = zeros(27,1);
mean_svm_weights = zeros(27,1);

%--------------------------------------------------------------------

% Set number of simulation runs,
% the classifier typically require ~100 runs to show nice weights,
% while a single run is sufficient for CCA
number_of_runs = 100;

% Set the general noise level,
% note that the processing time for the classifier increases significantly
% for higher noise levels, while it is constant for GLM and CCA


%general_noise_level = 0.1; % low
general_noise_level = 0.2; % high

activity_strength = 0.25;
activity_noise_level = 0.1;

% Perform analysis with GLM, CCA and SVM for a number of runs
% and average weights and other results
tic
for run = 1:number_of_runs
    
    run
    
    % Generate simulated test data for 4 "subjects",
    % each dataset has 3 x 3 x 3 voxels and N timepoints
    test_data1 = general_noise_level*randn(3,3,3,N);
    test_data2 = general_noise_level*randn(3,3,3,N);
    test_data3 = general_noise_level*randn(3,3,3,N);
    test_data4 = general_noise_level*randn(3,3,3,N);
    
    
    % Add positive activity in form of a 3D x
    test_data1(1,1,1,:) = squeeze(test_data1(1,1,1,:)) + activity_strength .* activity + activity_noise_level*randn(size(activity));
    test_data1(1,3,1,:) = squeeze(test_data1(1,3,1,:)) + activity_strength .* activity + activity_noise_level*randn(size(activity));
    test_data1(3,1,1,:) = squeeze(test_data1(3,1,1,:)) + activity_strength .* activity + activity_noise_level*randn(size(activity));
    test_data1(3,3,1,:) = squeeze(test_data1(3,3,1,:)) + activity_strength .* activity + activity_noise_level*randn(size(activity));
    test_data1(2,2,2,:) = squeeze(test_data1(2,2,2,:)) + activity_strength .* activity + activity_noise_level*randn(size(activity));
    test_data1(1,1,3,:) = squeeze(test_data1(1,1,3,:)) + activity_strength .* activity + activity_noise_level*randn(size(activity));
    test_data1(1,3,3,:) = squeeze(test_data1(1,3,3,:)) + activity_strength .* activity + activity_noise_level*randn(size(activity));
    test_data1(3,1,3,:) = squeeze(test_data1(3,1,3,:)) + activity_strength .* activity + activity_noise_level*randn(size(activity));
    test_data1(3,3,3,:) = squeeze(test_data1(3,3,3,:)) + activity_strength .* activity + activity_noise_level*randn(size(activity));
    
    test_data2(1,1,1,:) = squeeze(test_data2(1,1,1,:)) + activity_strength .* activity + activity_noise_level*randn(size(activity));
    test_data2(1,3,1,:) = squeeze(test_data2(1,3,1,:)) + activity_strength .* activity + activity_noise_level*randn(size(activity));
    test_data2(3,1,1,:) = squeeze(test_data2(3,1,1,:)) + activity_strength .* activity + activity_noise_level*randn(size(activity));
    test_data2(3,3,1,:) = squeeze(test_data2(3,3,1,:)) + activity_strength .* activity + activity_noise_level*randn(size(activity));
    test_data2(2,2,2,:) = squeeze(test_data2(2,2,2,:)) + activity_strength .* activity + activity_noise_level*randn(size(activity));
    test_data2(1,1,3,:) = squeeze(test_data2(1,1,3,:)) + activity_strength .* activity + activity_noise_level*randn(size(activity));
    test_data2(1,3,3,:) = squeeze(test_data2(1,3,3,:)) + activity_strength .* activity + activity_noise_level*randn(size(activity));
    test_data2(3,1,3,:) = squeeze(test_data2(3,1,3,:)) + activity_strength .* activity + activity_noise_level*randn(size(activity));
    test_data2(3,3,3,:) = squeeze(test_data2(3,3,3,:)) + activity_strength .* activity + activity_noise_level*randn(size(activity));
    
    % Add negative activity in form of a 3D x
    test_data3(1,1,1,:) = squeeze(test_data3(1,1,1,:)) - activity_strength .* activity + activity_noise_level*randn(size(activity));
    test_data3(1,3,1,:) = squeeze(test_data3(1,3,1,:)) - activity_strength .* activity + activity_noise_level*randn(size(activity));
    test_data3(3,1,1,:) = squeeze(test_data3(3,1,1,:)) - activity_strength .* activity + activity_noise_level*randn(size(activity));
    test_data3(3,3,1,:) = squeeze(test_data3(3,3,1,:)) - activity_strength .* activity + activity_noise_level*randn(size(activity));
    test_data3(2,2,2,:) = squeeze(test_data3(2,2,2,:)) - activity_strength .* activity + activity_noise_level*randn(size(activity));
    test_data3(1,1,3,:) = squeeze(test_data3(1,1,3,:)) - activity_strength .* activity + activity_noise_level*randn(size(activity));
    test_data3(1,3,3,:) = squeeze(test_data3(1,3,3,:)) - activity_strength .* activity + activity_noise_level*randn(size(activity));
    test_data3(3,1,3,:) = squeeze(test_data3(3,1,3,:)) - activity_strength .* activity + activity_noise_level*randn(size(activity));
    test_data3(3,3,3,:) = squeeze(test_data3(3,3,3,:)) - activity_strength .* activity + activity_noise_level*randn(size(activity));
    
    test_data4(1,1,1,:) = squeeze(test_data4(1,1,1,:)) - activity_strength .* activity + activity_noise_level*randn(size(activity));
    test_data4(1,3,1,:) = squeeze(test_data4(1,3,1,:)) - activity_strength .* activity + activity_noise_level*randn(size(activity));
    test_data4(3,1,1,:) = squeeze(test_data4(3,1,1,:)) - activity_strength .* activity + activity_noise_level*randn(size(activity));
    test_data4(3,3,1,:) = squeeze(test_data4(3,3,1,:)) - activity_strength .* activity + activity_noise_level*randn(size(activity));
    test_data4(2,2,2,:) = squeeze(test_data4(2,2,2,:)) - activity_strength .* activity + activity_noise_level*randn(size(activity));
    test_data4(1,1,3,:) = squeeze(test_data4(1,1,3,:)) - activity_strength .* activity + activity_noise_level*randn(size(activity));
    test_data4(1,3,3,:) = squeeze(test_data4(1,3,3,:)) - activity_strength .* activity + activity_noise_level*randn(size(activity));
    test_data4(3,1,3,:) = squeeze(test_data4(3,1,3,:)) - activity_strength .* activity + activity_noise_level*randn(size(activity));
    test_data4(3,3,3,:) = squeeze(test_data4(3,3,3,:)) - activity_strength .* activity + activity_noise_level*randn(size(activity));
    
    %---------------------------------------------------------
    % GLM
    %---------------------------------------------------------
    
    % Perform the analysis for the center voxel with the GLM
    timeseries = squeeze(test_data1(2,2,2,:));
    % Calculate beta weights
    beta1 = inv(X_GLM'*X_GLM)*X_GLM'*timeseries;
    % Calculate residuals
    e = timeseries' - beta1'*X_GLM';
    % Calculate t-test value
    ttest_GLM1 = beta1(1) / sqrt(var(e) * ctxtx_GLM);
    ttests_GLM1(run) = ttest_GLM1;
    betas_glm1(run) = beta1(1);
    
    timeseries = squeeze(test_data2(2,2,2,:));
    % Calculate beta weights
    beta2 = inv(X_GLM'*X_GLM)*X_GLM'*timeseries;
    % Calculate residuals
    e = timeseries' - beta2'*X_GLM';
    % Calculate t-test value
    ttest_GLM2 = beta2(1) / sqrt(var(e) * ctxtx_GLM);
    ttests_GLM2(run) = ttest_GLM2;
    betas_glm2(run) = beta2(1);
    
    timeseries = squeeze(test_data3(2,2,2,:));
    % Calculate beta weights
    beta3 = inv(X_GLM'*X_GLM)*X_GLM'*timeseries;
    % Calculate residuals
    e = timeseries' - beta3'*X_GLM';
    % Calculate t-test value
    ttest_GLM3 = beta3(1) / sqrt(var(e) * ctxtx_GLM);
    ttests_GLM3(run) = ttest_GLM3;
    betas_glm3(run) = beta3(1);
    
    timeseries = squeeze(test_data4(2,2,2,:));
    % Calculate beta weights
    beta4 = inv(X_GLM'*X_GLM)*X_GLM'*timeseries;
    % Calculate residuals
    e = timeseries' - beta4'*X_GLM';
    % Calculate t-test value
    ttest_GLM4 = beta4(1) / sqrt(var(e) * ctxtx_GLM);
    ttests_GLM4(run) = ttest_GLM4;
    betas_glm4(run) = beta4(1);
    
    mean_ttest_GLM = mean_ttest_GLM + (ttest_GLM1 + ttest_GLM2 + ttest_GLM3 + ttest_GLM4)/4;
    mean_beta_GLM = mean_beta_GLM + (beta1(1) + beta2(1) + beta3(1) + beta4(1))/4;
    
    %---------------------------------------------------------
    % CCA
    %---------------------------------------------------------
    
    
    % Perform the analysis for the center voxel with voxel-CCA
    % (we would normally use filter-CCA but it is quite hard to create a filter
    % with only 3 x 3 x 3 voxels)
    
    % Put the data into a matrix of N x 27 samples
    y = zeros(N,27);
    for i = 1:N
        vector = test_data1(:,:,:,i);
        y(i,:) = vector(:);
    end
    % Calculate covariance matrices
    C = cov([X_GLM y]);
    Cxx = C(1:2,1:2);
    Cyy = C(3:end,3:end);
    Cxy = C(1:2,3:end);
    Cyx = Cxy';
    % Solve eigenvalue problem(s)
    [eigv,eigs] = eig(C);
    % Get eigenvector corresponding to largest eigenvalue
    weights = eigv(:,29);
    % Divide eigenvector into beta and gamma (temporal and spatial weights)
    beta = weights(1:2);
    beta = beta/norm(beta);
    gamma = weights(3:end);
    gamma = gamma/norm(gamma);
    % Calculate canonical correlation
    corr1 = beta'*Cxy*gamma/sqrt(beta'*Cxx*beta * gamma'*Cyy*gamma);
    % The sign of the canonical correlation is not obvious,
    % since there are two sets of variables,
    % here we set the sign to be the sign of the weight for the first
    % temporal regressor (which describes the experimental paradigm)
    corr1 = corr1*sign(beta(1));
    canon_corrs1(run) = corr1;
    betas_cca1(run) = beta(1);
    mean_beta_CCA = mean_beta_CCA + beta(1)/4;
    mean_gamma = mean_gamma + gamma;
    
    %----------------------------------------
    
    % Put the data into a matrix of N x 27 samples
    y = zeros(N,27);
    for i = 1:N
        vector = test_data2(:,:,:,i);
        y(i,:) = vector(:);
    end
    % Calculate covariance matrices
    C = cov([X_GLM y]);
    Cxx = C(1:2,1:2);
    Cyy = C(3:end,3:end);
    Cxy = C(1:2,3:end);
    Cyx = Cxy';
    % Solve eigenvalue problem(s)
    [eigv,eigs] = eig(C);
    % Get eigenvector corresponding to largest eigenvalue
    weights = eigv(:,29);
    % Divide eigenvector into beta and gamma (temporal and spatial weights)
    beta = weights(1:2);
    beta = beta/norm(beta);
    gamma = weights(3:end);
    gamma = gamma/norm(gamma);
    % Calculate canonical correlation
    corr22 = beta'*Cxy*gamma/sqrt(beta'*Cxx*beta * gamma'*Cyy*gamma);
    % The sign of the canonical correlation is not obvious,
    % since there are two sets of variables,
    % here we set the sign to be the sign of the weight for the first
    % temporal regressor (which describes the experimental paradigm)
    corr22 = corr22*sign(beta(1));
    canon_corrs2(run) = corr22;
    betas_cca2(run) = beta(1);
    mean_beta_CCA = mean_beta_CCA + beta(1)/4;
    mean_gamma = mean_gamma + gamma;
    
    %----------------------------------------
    
    % Put the data into a matrix of N x 27 samples
    y = zeros(N,27);
    for i = 1:N
        vector = test_data3(:,:,:,i);
        y(i,:) = vector(:);
    end
    % Calculate covariance matrices
    C = cov([X_GLM y]);
    Cxx = C(1:2,1:2);
    Cyy = C(3:end,3:end);
    Cxy = C(1:2,3:end);
    Cyx = Cxy';
    % Solve eigenvalue problem(s)
    [eigv,eigs] = eig(C);
    % Get eigenvector corresponding to largest eigenvalue
    weights = eigv(:,29);
    % Divide eigenvector into beta and gamma (temporal and spatial weights)
    beta = weights(1:2);
    beta = beta/norm(beta);
    gamma = weights(3:end);
    gamma = gamma/norm(gamma);
    % Calculate canonical correlation
    corr3 = beta'*Cxy*gamma/sqrt(beta'*Cxx*beta * gamma'*Cyy*gamma);
    % The sign of the canonical correlation is not obvious,
    % since there are two sets of variables,
    % here we set the sign to be the sign of the weight for the first
    % temporal regressor (which describes the experimental paradigm)
    corr3 = corr3*sign(beta(1));
    canon_corrs3(run) = corr3;
    betas_cca3(run) = beta(1);
    mean_beta_CCA = mean_beta_CCA + beta(1)/4;
    
    %----------------------------------------
    
    % Put the data into a matrix of N x 27 samples
    y = zeros(N,27);
    for i = 1:N
        vector = test_data4(:,:,:,i);
        y(i,:) = vector(:);
    end
    % Calculate covariance matrices
    C = cov([X_GLM y]);
    Cxx = C(1:2,1:2);
    Cyy = C(3:end,3:end);
    Cxy = C(1:2,3:end);
    Cyx = Cxy';
    % Solve eigenvalue problem(s)
    [eigv,eigs] = eig(C);
    % Get eigenvector corresponding to largest eigenvalue
    weights = eigv(:,29);
    % Divide eigenvector into beta and gamma (temporal and spatial weights)
    beta = weights(1:2);
    beta = beta/norm(beta);
    gamma = weights(3:end);
    gamma = gamma/norm(gamma);
    % Calculate canonical correlation
    corr4 = beta'*Cxy*gamma/sqrt(beta'*Cxx*beta * gamma'*Cyy*gamma);
    % The sign of the canonical correlation is not obvious,
    % since there are two sets of variables,
    % here we set the sign to be the sign of the weight for the first
    % temporal regressor (which describes the experimental paradigm)
    corr4 = corr4*sign(beta(1));
    canon_corrs4(run) = corr4;
    betas_cca4(run) = beta(1);
    mean_beta_CCA = mean_beta_CCA + beta(1)/4;
    mean_gamma = mean_gamma + gamma;
    
    mean_canon_corr = mean_canon_corr + (corr1 + corr22 + corr3 + corr4)/4;
    
    %---------------------------------------------------------
    % CBA with SVM
    %---------------------------------------------------------
    
    % Perform the analysis for the center voxel with classification based
    % searchlight, using built-in SVM in Matlab
    
    try
        
        % Setup training and testing labels
        training = zeros(N/2,27);
        testing = zeros(N/2,27);
        for i = 1:N/2
            vector = test_data1(:,:,:,i);
            training(i,:) = vector(:);
        end
        for i = 1:N/2
            vector = test_data1(:,:,:,i+N/2);
            testing(i,:) = vector(:);
        end
        
        % Shift labels to adjust for BOLD delay
        training_labels = paradigm(1:N/2);
        temp = training_labels(1:3);
        training_labels(1:3) = training_labels(end-2:end);
        training_labels(4:end) = [temp ; training_labels(5:end-2)];
        
        testing_labels = paradigm(N/2+1:end);
        testing_labels(1:3) = testing_labels(end-2:end);
        testing_labels(4:end) = [temp ; testing_labels(5:end-2)];
        
        % Train SVM classifier
        svmstruct = svmtrain(training,training_labels);
        % Test classifier
        classifications = svmclassify(svmstruct,testing);
        % Calculate classification performance
        classification_performance1 = sum(classifications == testing_labels)/length(testing_labels);
        classification_performances1(run) = classification_performance1;
        
        % Calculate SVM weights
        svm_weights = zeros(27,1);
        for i = 1:27
            svm_weights(i) = sum(svmstruct.Alpha .* svmstruct.GroupNames(svmstruct.SupportVectorIndices) .* svmstruct.SupportVectors(:,i));
        end
        mean_svm_weights = mean_svm_weights + svm_weights;
        
        %----------------------------------------
        
        % Setup training and testing labels
        training = zeros(N/2,27);
        testing = zeros(N/2,27);
        for i = 1:N/2
            vector = test_data2(:,:,:,i);
            training(i,:) = vector(:);
        end
        for i = 1:N/2
            vector = test_data2(:,:,:,i+N/2);
            testing(i,:) = vector(:);
        end
        
        % Shift labels to adjust for BOLD delay
        training_labels = paradigm(1:N/2);
        temp = training_labels(1:3);
        training_labels(1:3) = training_labels(end-2:end);
        training_labels(4:end) = [temp ; training_labels(5:end-2)];
        
        testing_labels = paradigm(N/2+1:end);
        testing_labels(1:3) = testing_labels(end-2:end);
        testing_labels(4:end) = [temp ; testing_labels(5:end-2)];
        
        % Train SVM classifier
        svmstruct = svmtrain(training,training_labels);
        % Test classifier
        classifications = svmclassify(svmstruct,testing);
        % Calculate classification performance
        classification_performance2 = sum(classifications == testing_labels)/length(testing_labels);
        classification_performances2(run) = classification_performance2;
        
        % Calculate SVM weights
        svm_weights = zeros(27,1);
        for i = 1:27
            svm_weights(i) = sum(svmstruct.Alpha .* svmstruct.GroupNames(svmstruct.SupportVectorIndices) .* svmstruct.SupportVectors(:,i));
        end
        mean_svm_weights = mean_svm_weights + svm_weights;
        
        %----------------------------------------
        
        % Setup training and testing labels
        training = zeros(N/2,27);
        testing = zeros(N/2,27);
        for i = 1:N/2
            vector = test_data3(:,:,:,i);
            training(i,:) = vector(:);
        end
        for i = 1:N/2
            vector = test_data3(:,:,:,i+N/2);
            testing(i,:) = vector(:);
        end
        
        % Shift labels to adjust for BOLD delay
        training_labels = paradigm(1:N/2);
        temp = training_labels(1:3);
        training_labels(1:3) = training_labels(end-2:end);
        training_labels(4:end) = [temp ; training_labels(5:end-2)];
        
        testing_labels = paradigm(N/2+1:end);
        testing_labels(1:3) = testing_labels(end-2:end);
        testing_labels(4:end) = [temp ; testing_labels(5:end-2)];
        
        % Train SVM classifier
        svmstruct = svmtrain(training,training_labels);
        % Test classifier
        classifications = svmclassify(svmstruct,testing);
        % Calculate classification performance
        classification_performance3 = sum(classifications == testing_labels)/length(testing_labels);
        classification_performances3(run) = classification_performance3;
        
        %----------------------------------------
        
        % Setup training and testing labels
        training = zeros(N/2,27);
        testing = zeros(N/2,27);
        for i = 1:N/2
            vector = test_data4(:,:,:,i);
            training(i,:) = vector(:);
        end
        for i = 1:N/2
            vector = test_data4(:,:,:,i+N/2);
            testing(i,:) = vector(:);
        end
        
        % Shift labels to adjust for BOLD delay
        training_labels = paradigm(1:N/2);
        temp = training_labels(1:3);
        training_labels(1:3) = training_labels(end-2:end);
        training_labels(4:end) = [temp ; training_labels(5:end-2)];
        
        testing_labels = paradigm(N/2+1:end);
        testing_labels(1:3) = testing_labels(end-2:end);
        testing_labels(4:end) = [temp ; testing_labels(5:end-2)];
        
        % Train SVM classifier
        svmstruct = svmtrain(training,training_labels);
        % Test classifier
        classifications = svmclassify(svmstruct,testing);
        % Calculate classification performance
        classification_performance4 = sum(classifications == testing_labels)/length(testing_labels);
        classification_performances4(run) = classification_performance4;
        
    end
    
    mean_classification_performance = mean_classification_performance + (classification_performance1 + classification_performance2 + classification_performance3 + classification_performance4)/4;
    
end


figure
plot(squeeze(test_data1(2,2,2,:)))
hold on
plot([training_labels/2; testing_labels/2],'r')
hold off
legend('fMRI timeseries','Classifier labels')
title('One simulated fMRI timeseries with activity and the classifier labels')

mean_ttest_GLM = mean_ttest_GLM / number_of_runs
mean_beta_GLM = mean_beta_GLM / number_of_runs
mean_beta_CCA = mean_beta_CCA / number_of_runs
mean_canon_corr = mean_canon_corr / number_of_runs
mean_classification_performance = mean_classification_performance /  number_of_runs

% Reshape mean weights from a vector to a 3 x 3 x 3 cube
mean_svm_weights = reshape(mean_svm_weights,3,3,3)
mean_gamma = reshape(mean_gamma,3,3,3)

% Create true pattern
true_pattern = zeros(3,3,3);
true_pattern(1,1,1) = 1;
true_pattern(1,3,1) = 1;
true_pattern(3,1,1) = 1;
true_pattern(3,3,1) = 1;
true_pattern(2,2,2) = 1;
true_pattern(1,1,3) = 1;
true_pattern(1,3,3) = 1;
true_pattern(3,1,3) = 1;
true_pattern(3,3,3) = 1;
true_pattern

figure;
subplot(3,1,1); image(100*[true_pattern(:,:,1)/norm(true_pattern(:)) zeros(3,3) abs(mean_svm_weights(:,:,1))/norm(mean_svm_weights(:)) zeros(3,3) abs(mean_gamma(:,:,1))/norm(mean_gamma(:)) ]); title(sprintf('Patterns \n \n True                                                  SVM                                                CCA          ')); axis off
subplot(3,1,2); image(100*[true_pattern(:,:,2)/norm(true_pattern(:)) zeros(3,3) abs(mean_svm_weights(:,:,2))/norm(mean_svm_weights(:)) zeros(3,3) abs(mean_gamma(:,:,2))/norm(mean_gamma(:)) ]); axis off;
subplot(3,1,3); image(100*[true_pattern(:,:,3)/norm(true_pattern(:)) zeros(3,3) abs(mean_svm_weights(:,:,3))/norm(mean_svm_weights(:)) zeros(3,3) abs(mean_gamma(:,:,3))/norm(mean_gamma(:)) ]); axis off;

%print -dpng patterns_cba_vs_cca100.png
%print -depsc patterns_cba_vs_cca100.eps

figure
plot([ttests_GLM1 ttests_GLM2 ttests_GLM3 ttests_GLM4])
hold on
plot(5*[canon_corrs1 canon_corrs2 canon_corrs3 canon_corrs4],'r')
hold on
plot(100*[classification_performances1 classification_performances2 classification_performances3 classification_performances4 ],'g')
hold off
legend('t-test value','Canonical correlation','Classification performance')
title(sprintf('Activation values for GLM, CCA and CBA for 4 subjects and 100 runs per subject.'))
xlabel('Runs & Subjects')
ylabel('Activity value')

%print -dpng activation_glm_vs_cca_vs_cba.png
%print -depsc activation_glm_vs_cca_vs_cba.eps

%-------------------------------------------------------------------------------------------

cca_time = 0;
cba_time = 0;
parfor test = 1:10
    % Measure processing time for CCA
    
    y = zeros(N,27);
    for i = 1:N
        vector = test_data1(:,:,:,i);
        y(i,:) = vector(:);
    end
    % Calculate covariance matrices
    start = clock;
    C = cov([X_GLM y]);
    Cxx = C(1:2,1:2);
    Cyy = C(3:end,3:end);
    Cxy = C(1:2,3:end);
    Cyx = Cxy';
    % Solve eigenvalue problem(s)
    [eigv,eigs] = eig(C);
    % Get eigenvector corresponding to largest eigenvalue
    weights = eigv(:,29);
    % Divide eigenvector into beta and gamma (temporal and spatial weights)
    beta = weights(1:2);
    beta = beta/norm(beta);
    gamma = weights(3:end);
    gamma = gamma/norm(gamma);
    % Calculate canonical correlation
    corr4 = beta'*Cxy*gamma/sqrt(beta'*Cxx*beta * gamma'*Cyy*gamma);
    cca_time = cca_time + etime(clock,start);
    
    % Measure processing time for CBA
    training = zeros(N/2,27);
    testing = zeros(N/2,27);
    for i = 1:N/2
        vector = test_data1(:,:,:,i);
        training(i,:) = vector(:);
    end
    for i = 1:N/2
        vector = test_data1(:,:,:,i+N/2);
        testing(i,:) = vector(:);
    end
    
    % Shift labels to adjust for BOLD delay
    training_labels = paradigm(1:N/2);
    temp = training_labels(1:3);
    training_labels(1:3) = training_labels(end-2:end);
    training_labels(4:end) = [temp ; training_labels(5:end-2)];
    
    testing_labels = paradigm(N/2+1:end);
    testing_labels(1:3) = testing_labels(end-2:end);
    testing_labels(4:end) = [temp ; testing_labels(5:end-2)];
    
    % Train SVM classifier
    start = clock;
    svmstruct = svmtrain(training,training_labels);
    % Test classifier
    classifications = svmclassify(svmstruct,testing);
    % Calculate classification performance
    classification_performance = sum(classifications == testing_labels)/length(testing_labels);
    cba_time = cba_time + etime(clock,start);
    
end


% Measure difference between CCA and CBA
cca_speedup = cba_time / cca_time


