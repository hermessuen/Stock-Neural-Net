% This script will create, train , and save the time series neural network
clc ; clear

listOfStocks = ["F"; "GM"; "HOG"; "IRBT"; "WHR"; "KORS"; "EXPR"...
    ;"CAKE";"DPZ"; "ULTA"; "FIVE"; "CASY"; "KR"...
    ;"BUD"; "PM"; "ISRG"; "JAZZ"; "ARNA"; "BIO"; "BOFI"; "FITB"];

outputSize = 'full'; %compact returns last 20, full is 20 years of data
typeData = 'TIME_SERIES_MONTHLY_ADJUSTED';  %Options include:daily, weekly, monthly, and adjusted for all

numStocks = length(listOfStocks);

XTrainMaster = cell(numStocks,1);
YTrainMaster = cell(numStocks,1);

home = pwd;
for i = 1:numStocks
    symbol  = listOfStocks(i); %The stock ticker symbol
    nameOfXTrain = strcat(home,'/DATA/',symbol,'XTrain.mat');
    nameOfYTrain = strcat(home, '/DATA/', symbol, 'YTrain.mat');
    if ~(exist(nameOfXTrain, 'file') == 2 && exist(nameOfYTrain,'file') == 2)
        fprintf('Waiting %d seconds before querying database...\n', 10);
        pause(1);
        %Getting the stock data if it did not exist
        getStockData(symbol, outputSize, typeData); 
    end
    
    if exist(nameOfXTrain, 'file') == 2 && exist(nameOfYTrain,'file') == 2
        fprintf('Received data for stock: %s\n', symbol);
        load(nameOfXTrain);
        load(nameOfYTrain);
        
        % only get the normalized data
        XTrain = XTrain(6:10, :);
        YTrain = YTrain(8,:);
        XTrainMaster{i} = XTrain;
        YTrainMaster{i} = YTrain;
    else
        fprintf('Could not get data for stock %s for training \n', symbol);
    end
end

% Prepare data for padding here
sequenceLengths = zeros(1,numel(XTrainMaster));
for i =1:numel(XTrainMaster)
    sequence = XTrainMaster{i};
    sequenceLengths(i) = size(sequence,2);
end

[sequenceLengthsSorted, idx] = sort(sequenceLengths, 'descend');
XTrainMaster = XTrainMaster(idx);
YTrainMaster = YTrainMaster(idx);




layer1 = sequenceInputLayer(10);
layer2 = bilstmLayer(512, 'OutputMode', 'sequence');
layer3 = bilstmLayer(512, 'OutputMode', 'sequence');
layer4 = fullyConnectedLayer(1);
layer5 = regressionLayer();


layers = [layer1;layer2;layer3;layer4;layer5];
options = trainingOptions('adam',...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.2,...
    'LearnRateDropPeriod',125,...
    'MaxEpochs',200,...
    'MiniBatchSize',3,...
    'Plots','training-progress','InitialLearnRate', 0.0001, 'ExecutionEnvironment', ...
    'auto', 'Shuffle', 'never', 'GradientThreshold', 1, 'Verbose', 0);

fprintf('Training net...\n');
net = trainNetwork(XTrainMaster,YTrainMaster, layers, options);
save('trainedNet', 'net', '-v6');






