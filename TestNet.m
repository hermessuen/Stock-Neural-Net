%% This will test the accuracy of the neural network

if ~(exist('trainedNet.mat', 'file') == 2) 
   fprintf('NETWORK DOES NOT EXIST \n');

else
    load('trainedNet.mat');
    % the stocks we want to test
    listOfStocks = ["F"; "GM"; "HOG"; "IRBT"; "WHR"; "KORS"; "EXPR"...
        ;"CAKE";"DPZ"; "ULTA"; "FIVE"; "CASY"; "KR"...
        ;"BUD"; "PM"; "ISRG"; "JAZZ"; "ARNA"; "BIO"; "BOFI"; "FITB"];
    
    outputSize = 'full'; %compact returns last 20, full is 20 years of data
    typeData = 'TIME_SERIES_MONTHLY_ADJUSTED';  %Options include:daily, weekly, monthly, and adjusted for all
    
    numStocks = length(listOfStocks);
    
    predictions = zeros(1,numStocks);
    actual = zeros(1,numStocks);
    error = zeros(1,numStocks);
    
    home = pwd;
    for i = 1:numStocks
        symbol  = listOfStocks(i); %The stock ticker symbol
        nameOfXTest = strcat(home,'/DATA/',symbol,'XTest.mat');
        nameOfYTest = strcat(home, '/DATA/', symbol, 'YTest.mat');
        if ~(exist(nameOfXTest, 'file') == 2 && exist(nameOfYTest,'file') == 2)
            fprintf('Waiting %d seconds before querying database...\n', 10);
            pause(1);
            % Get the stock data if we have not downloaded it previously
            getStockData(symbol, outputSize, typeData); %Getting the stock data if it did not exist
        end
        
        if exist(nameOfXTest, 'file') == 2 && exist(nameOfYTest,'file') == 2
            fprintf('Received data for stock: %s\n', symbol);
            load(nameOfXTest);
            load(nameOfYTest);
            % Extract the normalized values
            XTest = XTest(6:10,:);
            YTest = YTest(8,:);
       
        else
            fprintf('Could not get data for stock %s \n', symbol);
        end
        % use the net to make the prediction 
        [updatedNet,prediction] = predictAndUpdateState(net, XTest);
        % we only need the last time step
        prediction = prediction(end);
        YTest = YTest(end);
        % keeping track of the "sign" of the return
        % 1 is positive, 0 is negative
        predictions(i) = prediction > 0;
        actual(i) = YTest > 0;
        
        error(i) = prediction - YTest;
    end
    % get the root mean squared error for all of the stocks
    error = sqrt(mean(error.^2));
    
    % see the percentage of the stocks that the model is able to predict
    % the "sign" of the return
    numCorrect = nnz(predictions == actual);
    accuracy = numCorrect/numel(predictions); %anything over 50% will be 
                                               % considered a "good" model
end









