%% This function will save the stock data for a given company.
%  Saves XTrain, YTrain, XTest, and YTest to a local folder: "DATA".
% Arguments: 
%   symbol:The Stock Ticker
%   outputSize: compact or full. Compact 100 data points, full is 20 years
%   typeData: what data you want: daily, monthly, adjusted, 


function getStockData(symbol, outputSize, typeData)
%% Define the parameters for the query
baseUrl = 'https://www.alphavantage.co/query?';
apiKey = 'GP4AEWI9SXWI95VN';


%% build the url and call the API

   
url = strcat(baseUrl , 'outputsize=' , outputSize , '&function=' , typeData , ...
    '&symbol=' , symbol , '&apikey=' , apiKey);

options = weboptions('Timeout', 50);

try
    data = webread(url, options);



%% Create Predictor Matrices

% The MAT file must contain a time series represented by a matrix with 
% rows corresponding to data points, and columns corresponding to time steps.
dataPoints = fieldnames(data.MonthlyAdjustedTimeSeries);
dataPoints = flipud(dataPoints); % Get the data into chronological order
numDataPoints = length(dataPoints);
data = data.MonthlyAdjustedTimeSeries;




%Use 0.95 of available data for training and put the remaining 0.05 for
%testing
dataPointsForTrain = round(0.95*numDataPoints,0);

%Pre-allocate data for the mat file
XTrain = zeros(10,numDataPoints);

fprintf('Formatting Data for %s for training... \n', symbol);
for i = 1:numDataPoints
    dataPoint = getfield(data,char(dataPoints(i)));
    XTrain(1,i) = str2double(dataPoint.x1_Open);
    XTrain(2,i) = str2double(dataPoint.x2_High);
    XTrain(3,i) = str2double(dataPoint.x3_Low);
    XTrain(4,i) = str2double(dataPoint.x4_Close);
    XTrain(5,i) = str2double(dataPoint.x5_AdjustedClose);
    XTrain(6,i) = str2double(dataPoint.x6_Volume)/1000000; %Scaled Volume
    XTrain(7,i) = (XTrain(2,i) - XTrain(3,i))/XTrain(4,i);  %(High-Low)/Close
    
    %only calculate these statistics if possible
    if i ~=1
        XTrain(8,i) = (XTrain(5,i) - XTrain(5,i-1))/XTrain(5,i-1); % (CurrentClose - prevClose)/prevClose
        XTrain(9,i) = (XTrain(6,i) - XTrain(6,i-1))/XTrain(6,i-1); %(Volume - prevVolume)/prevVolume
    end
    
    if i > 12
        XTrain(10,i) = (XTrain(5,i-1) - XTrain(5,i-12))/XTrain(5,i-12); %(LastMonthClose - LastYearClose)/LastYearClose
    end
end

XTest = XTrain(:,dataPointsForTrain+1:numDataPoints);
XTrain = XTrain(:,1:dataPointsForTrain);
len = size(XTrain);

%Trim out the padded zero data
if len(2) > 12
    XTrain = XTrain(:,13:end);
end


%% Create Response Matrices

YTrain = XTrain(:,2:end);
YTest = XTest(:,2:end);

XTrain = XTrain(:,1:end-1);
XTest = XTest(:,1:end-1);


home = pwd;
symbolXTrain = strcat(home, '/DATA/',symbol, 'XTrain.mat');
symbolYTrain = strcat(home, '/DATA/',symbol, 'YTrain.mat');

symbolXTest = strcat(home, '/DATA/',symbol, 'XTest.mat');
symbolYTest = strcat(home, '/DATA/', symbol, 'YTest.mat');


%% Save The Predictor and Response Matrices to Mat Files
fprintf('Saving Data for %s for training... \n', symbol);
save(symbolXTrain, 'XTrain', '-v6');
save(symbolYTrain, 'YTrain', '-v6');

fprintf('Saving Data for %s for testing... \n', symbol);
save(symbolXTest, 'XTest', '-v6');
save(symbolYTest, 'YTest', '-v6');
fprintf('Done...\n');


catch
    fprintf('Failed getting data for stock: %s \n', symbol);
    return
end
end
