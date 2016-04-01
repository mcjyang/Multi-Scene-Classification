classdef HW3_BoW    
% Practical for Visual Bag-of-Words representation    
% Use SVM, instead of Least-Squares SVM as for MPP_BoW
% By: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
% Created: 18-Dec-2015
% Last modified: 18-Dec-2015    
    
    methods (Static)
        function main()
            scales = [8, 16, 32, 64];
            normH = 16;
            normW = 16;
            bowCs = HW3_BoW.learnDictionary(scales, normH, normW);

            load('kcenter.mat');
            bowCs = center';
            
            [trIds, trLbs] = ml_load('../bigbangtheory/train.mat',  'imIds', 'lbs');             
            tstIds = ml_load('../bigbangtheory/test.mat', 'imIds'); 
                        
            trD  = HW3_BoW.cmpFeatVecs(trIds, scales, normH, normW, bowCs);             tstD = HW3_BoW.cmpFeatVecs(tstIds, scales, normH, normW, bowCs);
%             save('Feat.mat','trD','tstD');
%             load('Feat.mat');

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Write code for training svm and prediction here            %
            
            % Cross validation for rbf kernel
%             accuracy = zeros(1,6);
%             for i=1:6
%                 string = ['-t 2 -v 5 -g 10 -c ',num2str(10^(i-1)),' -q'];
%                 c = num2str(10^(i-1))
%                 accuracy(i) = svmtrain(trLbs, trD',string);
%             end
            
%             fig1 = figure(1);
%             plot([1:6],accuracy/100,'-ro');
%             legend('accuracy versus c');
%             axis([1,6,0,1]);
%             ax=gca;
%             ax.XTickLabel = {'1','','10','','100','','1000','','10000','','100000'};
%             saveas(fig1,'accuracy_vs_c.png');
%             close();

            % rbf kernel prediction
            gamma = 10;
            c = 100;
            model_rbf = svmtrain(trLbs,trD','-t 2 -g 10 -c 100 -q');
            predict_label_rbf = svmpredict((1:1600)', tstD',model_rbf,'-q');

%             save('svm_rbf_1.mat','predict_label_rbf','model_rbf');
%             load('svm_rbf_1.mat');
            

%             Cross validation for self kernel
%             accuracy=zeros(1,30);
%             for i = 1 : 5
%                 gamma = 10^(i-3)
%                 c=10;
%                 [trainK testK] = cmpExpX2Kernel(trD', trD', gamma);
%                 accuracy(i) = svmtrain(trLbs,[(1:size(trainK,1))', trainK],['-q -t 4 -v 5 -c ',num2str(c)]);
%             end
            
            % rbf kernel prediction
            gamma = 0.5;
            c=100;
            [trainK testK] = cmpExpX2Kernel(trD', tstD', gamma);
            model_expX2 = svmtrain(trLbs,[(1:size(trainK,1))', trainK],['-t 4 -q -c ',num2str(c)]);
            predict_label_expX2 = svmpredict(predict_label_rbf,[(1:size(testK,1))',testK],model_expX2,'-q');
            

            ImgId=tstIds;
            Prediction=predict_label_expX2;
            result=table(ImgId,Prediction);
            writetable(result,'predTestLabels.csv');

            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
        end;
                
        function bowCs = learnDictionary(scales, normH, normW)
            % Number of random patches to build a visual dictionary
            % Should be around 1 million for a robust result
            % We set to a small number her to speed up the process. 
            nPatch2Sample = 100000;
            
            % load train ids
            trIds = ml_load('../bigbangtheory/train.mat', 'imIds'); 
            nPatchPerImScale = ceil(nPatch2Sample/length(trIds)/length(scales));
                        
            randWins = cell(length(scales), length(trIds)); % to store random patches
            for i=1:length(trIds)
                ml_progressBar(i, length(trIds), 'Randomly sample image patches');
                im = imread(sprintf('../bigbangtheory/%06d.jpg', trIds(i)));
                im = double(rgb2gray(im));  
                for j=1:length(scales)
                    scale = scales(j);
                    winSz = [scale, scale];
                    stepSz = winSz/2; % stepSz is set to half the window size here. 
                    
                    % ML_SlideWin is a class for efficient sliding window 
                    swObj = ML_SlideWin(im, winSz, stepSz);
                    
                    % Randomly sample some patches
                    randWins_ji = swObj.getRandomSamples(nPatchPerImScale);
                    
                    % resize all the patches to have a standard size
                    randWins_ji = reshape(randWins_ji, [scale, scale, size(randWins_ji,2)]);                    
                    randWins{j,i} = imresize(randWins_ji, [normH, normW]);
                end
            end;
            randWins = cat(3, randWins{:});
            randWins = reshape(randWins, [normH*normW, size(randWins,3)]);
                                    
            fprintf('Learn a visual dictionary using k-means\n');
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                       
            % Input: randWinds contains your data points                 
            % bowCs: centroids from k-means, one column for each centroid  
            
            
            random=1;
            k=1000;
            [~,center,~] = MyKmeans(randWins',k,random);
%             save('kcenter.mat','center');
%             load('kcenter.mat');
            bowCs = center';
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end;
                
        function D = cmpFeatVecs(imIds, scales, normH, normW, bowCs)
            n = length(imIds);
            D = cell(1, n);
            startT = tic;
            for i=1:n
                ml_progressBar(i, n, 'Computing feature vectors', startT);
                im = imread(sprintf('../bigbangtheory/%06d.jpg', imIds(i)));                                
                bowIds = HW3_BoW.cmpBowIds(im, scales, normH, normW, bowCs);                
                feat = hist(bowIds, 1:size(bowCs,2));
                D{i} = feat(:);
            end;
            D = cat(2, D{:});
            D = double(D);
            D = D./repmat(sum(D,1), size(D,1),1);
        end        
        
        % bowCs: d*k matrix, with d = normH*normW, k: number of clusters
        % scales: sizes to densely extract the patches. 
        % normH, normW: normalized height and width oMf patches
        function bowIds = cmpBowIds(im, scales, normH, normW, bowCs)
            im = double(rgb2gray(im));
            bowIds = cell(length(scales),1);
            for j=1:length(scales)
                scale = scales(j);
                winSz = [scale, scale];
                stepSz = winSz/2; % stepSz is set to half the window size here.
                
                % ML_SlideWin is a class for efficient sliding window
                swObj = ML_SlideWin(im, winSz, stepSz);
                nBatch = swObj.getNBatch();
                
                for u=1:nBatch
                    wins = swObj.getBatch(u);
                    
                    % resize all the patches to have a standard size
                    wins = reshape(wins, [scale, scale, size(wins,2)]);                    
                    wins = imresize(wins, [normH, normW]);
                    wins = reshape(wins, [normH*normW, size(wins,3)]);
                    
                    % Get squared distance between windows and centroids
                    dist2 = ml_sqrDist(bowCs, wins); % dist2: k*n matrix, 
                    
                    % bowId: is the index of cluster closest to a patch
                    [~, bowIds{j,u}] = min(dist2, [], 1);                     
                end                
            end;
            bowIds = cat(2, bowIds{:});
        end;        
        
    end    
end

