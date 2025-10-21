function results = CNN_LSTM_CV(X, k)
% CNN_LSTM_CV - Train CNN+LSTM hybrid with k-fold cross-validation
% Input: 
%   X - [samples x (features+1)] matrix, last column = label
%   k - number of folds
% Output:
%   results - struct with accuracy, sensitivity, specificity, Loss

    if nargin < 2, k = 5; 
    end

    features = X(:,1:end-1);
    labels = categorical(X(:,end));
    features = zscore(features);
    
    numFeatures = size(features,2);
    numClasses = numel(categories(labels));
    
    cv = cvpartition(labels,'KFold',k);

    acc_all = zeros(k,1);
    sens_all = zeros(k,1);
    spec_all = zeros(k,1);

    for i = 1:k
                fprintf('Fold %d/%d (CNN+LSTM)\n',i,k);
                % 1. Split train/test indices
                trainIdx = training(cv,i);
                testIdx  = test(cv,i);
                
                % 2. Extract train/test data
                XTrainMat = features(trainIdx, :); % numeric [samples x features]
                YTrain    = labels(trainIdx);      % categorical vector of same length
                XTestMat  = features(testIdx, :);
                YTest     = labels(testIdx);
                
                % 3. Convert train/test to cell arrays
                XTrainCell = arrayfun(@(j) XTrainMat(j,:)', 1:size(XTrainMat,1), 'UniformOutput', false);
                XTestCell  = arrayfun(@(j) XTestMat(j,:)',  1:size(XTestMat,1), 'UniformOutput', false);

                
                layers = [
                sequenceInputLayer(numFeatures,'Name','input')
                convolution1dLayer(5,32,'Padding','same','Name','conv1')
                batchNormalizationLayer('Name','bn1')
                reluLayer('Name','relu1')
                dropoutLayer(0.3,'Name','drop1')
                lstmLayer(32,'OutputMode','last','Name','lstm1')
                fullyConnectedLayer(numClasses,'Name','fc')
                softmaxLayer('Name','softmax')
                classificationLayer('Name','output')
                ];
                

                options = trainingOptions('adam', ...
                'MaxEpochs',40, ...
                'MiniBatchSize',16, ...
                'Shuffle','every-epoch', ...
                'Plots','none', ...              % disable MATLAB GUI, we’ll plot manually
                'Verbose',0, ...
                'ValidationFrequency',10, ...
                'ExecutionEnvironment','auto');


                
                net = trainNetwork(XTrainCell,YTrain,layers,options);
                YPred = classify(net,XTestCell);
                
                % ---- Train network and record training info ----
                [net, trainInfo] = trainNetwork(XTrainCell,YTrain,layers,options);
                
                % Store training accuracy and loss
                trainingAccuracy{i} = trainInfo.TrainingAccuracy;
                trainingLoss{i} = trainInfo.TrainingLoss;
                
                % ---- Evaluate test data ----
                YPred = classify(net,XTestCell);
                testAccuracy(i) = sum(YPred==YTest)/numel(YTest); % Compute test accuracy
                
                % Compute confusion matrix and other metrics
                confMat = confusionmat(YTest, YPred);
                
                
                
                % net = trainNetwork(XTrain,YTrain,layers,options);
                % 
                % YPred = classify(net,XTest);
                acc = sum(YPred==YTest)/numel(YTest);
                confMat = confusionmat(YTest, YPred);
                
                if numClasses == 2
                TP = confMat(2,2);
                TN = confMat(1,1);
                FP = confMat(1,2);
                FN = confMat(2,1);
                sens = TP / (TP + FN + eps);
                spec = TN / (TN + FP + eps);
                % else
                %     sens = mean(diag(confMat)./ (sum(confMat,2)+eps));
                %     spec = NaN;
                end
                % fprintf(acc);
                
                acc_all(i) = acc;
                sens_all(i) = sens;
                spec_all(i) = spec;
    end

results.accuracy = acc_all;
results.sensitivity = sens_all;
results.specificity = spec_all;
results.meanAccuracy = mean(acc_all);
results.meanSensitivity = mean(sens_all);
results.meanSpecificity = mean(spec_all(~isnan(spec_all)));

results.trainingAccuracy = trainingAccuracy;
results.trainingLoss = trainingLoss;
results.testAccuracy = testAccuracy; % Add test accuracy to results

fprintf('LSTM+GRU\n');
fprintf('Average Accuracy: %.2f%%\n', results.meanAccuracy*100);
fprintf('Average Sensitivity: %.2f%%\n', results.meanSensitivity*100);
fprintf('Average Specificity: %.2f%%\n', results.meanSpecificity*100);


end
