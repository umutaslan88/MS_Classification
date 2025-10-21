
clc; clear all; close all;


load('Dataset_Whole_30s.mat');

%% 


acc_LSTM_GRU =[];
sens_LSTM_GRU = [];
spec_LSTM_GRU = [];

acc_CNN_LSTM =[];
sens_CNN_LSTM = [];
spec_CNN_LSTM = [];

acc_MLP =[];
sens_MLP = [];
spec_MLP = [];
acc_all_CNN_LSTM = [];
acc_all_LSTM_GRU = [];
sens_all_CNN_LSTM = [];
sens_all_LSTM_GRU = [];
spec_all_CNN_LSTM = [];
spec_all_LSTM_GRU = [];
data = Dataset_Whole;
for i=1:3
        results1 = MLP_EEG_CV(data, 5);   % Implement MLP
        results2 = CNN_LSTM_CV(data, 5);  % Implement CNN  + LSTM  
        results3 = LSTM_GRU_CV(data, 5);  % Implement LSTM + GRU 
        
        %%%%   MLP   %%%%
        sd_MLP= struct2cell(results1);
        average=cell2mat(sd_MLP(4));
        sensi=cell2mat(sd_MLP(5));
        spec=cell2mat(sd_MLP(6));
        acc_MLP(i) =  average;
        sens_MLP(i) =  sensi;
        spec_MLP(i) =  spec;
        
        %%%%   CNN+ LSTM   %%%%        
        sd_CNN_LSTM= struct2cell(results2);
        average1=cell2mat(sd_CNN_LSTM(4));
        sensi1=cell2mat(sd_CNN_LSTM(5));
        spec1=cell2mat(sd_CNN_LSTM(6));
        acc_CNN_LSTM(i) =  average1;
        sens_CNN_LSTM(i) =  sensi1;
        spec_CNN_LSTM(i) =  spec1;
        
        %%%%   LSTM + GRU   %%%%        
        sd_LSTM_GRU= struct2cell(results3);
        average2=cell2mat(sd_LSTM_GRU(4));
        sensi2=cell2mat(sd_LSTM_GRU(5));
        spec2=cell2mat(sd_LSTM_GRU(6));
        acc_LSTM_GRU(i) =  average2;
        sens_LSTM_GRU(i) =  sensi2;
        spec_LSTM_GRU(i) =  spec2;
        
        
        acc_all_CNN_LSTM = [acc_all_CNN_LSTM; results2.accuracy(:)];
        acc_all_LSTM_GRU = [acc_all_LSTM_GRU; results3.accuracy(:)];
        sens_all_CNN_LSTM = [sens_all_CNN_LSTM; results2.sensitivity(:)];
        sens_all_LSTM_GRU = [sens_all_LSTM_GRU; results3.sensitivity(:)];
        spec_all_CNN_LSTM = [spec_all_CNN_LSTM; results2.specificity(:)];
        spec_all_LSTM_GRU = [spec_all_LSTM_GRU; results3.specificity(:)];

end

%% 


    fprintf('---- DL Methods---- \n');
    fprintf('  Accuracy CNN+LSTM:  %.2f ± %.2f %% \n', mean(acc_all_CNN_LSTM)*100, std(acc_all_CNN_LSTM)*100);
    fprintf('  Sens CNN+LSTM:   %.2f ± %.2f %% \n', mean(sens_all_CNN_LSTM)*100, std(sens_all_CNN_LSTM)*100);
    fprintf('  Spec CNN+LSTM:   %.2f ± %.2f %%  \n', mean(spec_all_CNN_LSTM)*100, std(spec_all_CNN_LSTM)*100);

    fprintf('  Accuracy LSTM+GRU:  %.2f ± %.2f %%\n' , mean(acc_all_LSTM_GRU)*100, std(acc_all_LSTM_GRU)*100);
    fprintf('  Sen LSTM+GRU:     %.2f ± %.2f %%\n', mean(sens_all_LSTM_GRU)*100, std(sens_all_LSTM_GRU)*100);
    fprintf('  Sepc LSTM+GRU:      %.2f ± %.2f %%\n', mean(spec_all_LSTM_GRU)*100, std(spec_all_LSTM_GRU)*100);



    


%% Machine Learning Methods


on= ones(288,1);
zer = zeros(220,1);
label= [on;zer];

X =data(:,1:end-1);

rng('default');  % For reproducibility
Y =Dataset_Whole(:,end);

% Inputs
X = double(X);       % Ensure features are double
Y = categorical(Y);  % Convert labels to categorical

k = 5;               % Number of folds
reps = 10;           % Number of repetitions

% Initialize results
methods = {'KNN', 'Decision Tree', 'Random Forest', 'Boosted Ensemble', 'Deep Learning'};
nMethods = numel(methods);
results = struct();

for m = 1:nMethods
    results(m).name = methods{m};
    results(m).acc = zeros(reps, 1);
    results(m).sens = zeros(reps, 1);
    results(m).spec = zeros(reps, 1);
end

for rep = 1:reps
    cv = cvpartition(Y, 'KFold', k);
    
    for fold = 1:k
        trainIdx = training(cv, fold);
        testIdx = test(cv, fold);

        Xtrain = X(trainIdx,:);
        Ytrain = Y(trainIdx);
        Xtest = X(testIdx,:);
        Ytest = Y(testIdx);

        % ---------- 1. KNN ----------
        mdlKNN = fitcknn(Xtrain, Ytrain, 'NumNeighbors', 5);
        predKNN = predict(mdlKNN, Xtest);

        % ---------- 2. Decision Tree ----------
        mdlDT = fitctree(Xtrain, Ytrain);
        predDT = predict(mdlDT, Xtest);

        % ---------- 3. Random Forest ----------
        mdlRF = TreeBagger(50, Xtrain, Ytrain, 'OOBPrediction', 'off');
        predRF = predict(mdlRF, Xtest);
        predRF = categorical(predRF);

        % ---------- 4. Boosted Ensemble ----------
        t = templateTree('MaxNumSplits', 10);
        mdlBoost = fitcensemble(Xtrain, Ytrain, 'Method', 'AdaBoostM1', ...
                                'NumLearningCycles', 50, 'Learners', t);
        predBoost = predict(mdlBoost, Xtest);

        % ---------- 5. Deep Learning (Shallow MLP) ----------
        layers = [
            featureInputLayer(size(Xtrain,2))
            fullyConnectedLayer(32)
            reluLayer
            fullyConnectedLayer(numel(categories(Y)))
            softmaxLayer
            classificationLayer];

        options = trainingOptions('adam', ...
            'MaxEpochs', 50, ...
            'MiniBatchSize', 32, ...
            'Verbose', false, ...
            'Shuffle', 'every-epoch');

        net = trainNetwork(Xtrain, Ytrain, layers, options);
        predDL = classify(net, Xtest);

        % Store results
        preds = {predKNN, predDT, predRF, predBoost, predDL};
        for m = 1:nMethods
            acc = mean(preds{m} == Ytest);

            % Confusion matrix
            cm = confusionmat(Ytest, preds{m});
            % Sensitivity = TP / (TP + FN)
            % Specificity = TN / (TN + FP)
            if size(cm,1) == 2
                TP = cm(1,1);
                FN = cm(1,2);
                FP = cm(2,1);
                TN = cm(2,2);
                sens = TP / (TP + FN + eps);
                spec = TN / (TN + FP + eps);
            else
                % For multi-class: macro average sensitivity & specificity
                sens = mean(diag(cm)./sum(cm,2));
                spec = mean((sum(cm(:)) - sum(cm,1)' - sum(cm,2) + diag(cm)) ./ ...
                            (sum(cm(:)) - sum(cm,2)));
            end

            results(m).acc(rep) = acc;
            results(m).sens(rep) = sens;
            results(m).spec(rep) = spec;
        end
    end
end

% --------- Print Results ---------
fprintf('\nClassification Results (mean ± std over %d repetitions):\n', reps);
for m = 1:nMethods
    fprintf('\n%s:\n', results(m).name);
    fprintf('  Accuracy:     %.2f ± %.2f %%\n', mean(results(m).acc)*100, std(results(m).acc)*100);
    fprintf('  Sensitivity:  %.2f ± %.2f %%\n', mean(results(m).sens)*100, std(results(m).sens)*100);
    fprintf('  Specificity:  %.2f ± %.2f %%\n', mean(results(m).spec)*100, std(results(m).spec)*100);
end
