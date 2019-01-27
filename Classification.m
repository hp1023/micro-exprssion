clear all;
close all;
clc;
%% read label
FileName = "..\\combined_3class_gt.csv";
FileFeature = "FeatureDecriptors_16";
FileTabel = readtable(FileName);
%%
train_num = 442;
Feature_num = 177*16;
Subject_num = FileTabel(1:train_num, 2);
Subject_num = tabulate(table2cell(Subject_num));% 统计测试人员编号
%%
Predict = zeros(train_num, 2);
pre = 1;
TP_0 = 0; TP_1 = 0; TP_2 = 0;
FP_0 = 0; FP_1 = 0; FP_2 = 0;
FN_0 = 0; FN_1 = 0; FN_2 = 0;
TN_0 = 0; TN_1 = 0; TN_2 = 0;

for i = 1 : length(Subject_num)
    Test_num = cell2mat(Subject_num(i,2));
    Train_num = train_num - Test_num;
        
    TrainFeature_src = zeros(Train_num, Feature_num);
    TrainLabel_src = zeros(Train_num, 1);
    k = 1;
       
    Test_Subject = cell2mat(Subject_num(i,1));
    TestFeaturePath = "..\\" + FileFeature + "\\SubjectFeatureMap_" + Test_Subject + ".mat";
    TestLabelPath = "..\\" + FileFeature + "\\SubjectLabelMap_" + Test_Subject + ".mat";
    load(TestFeaturePath);
    load(TestLabelPath);
    TestFeature = FeatureMap;
    TestLabel = LabelMap;
    fprintf("第%d轮训练的测试集：%s,测试集个数：%d\n", i, Test_Subject, Test_num);
    for j = 1 : length(Subject_num)
        Train_Subject = cell2mat(Subject_num(j,1));
        Train_Subject_num = cell2mat(Subject_num(j,2));
        if i ~= j
            TrainFeaturePath = "..\\" + FileFeature + "\\SubjectFeatureMap_" + Train_Subject + ".mat";
            TrainLabelPath = "..\\" + FileFeature + "\\SubjectLabelMap_" + Train_Subject + ".mat";
            load(TrainFeaturePath);
            load(TrainLabelPath);
            TrainFeature_src(k:k+Train_Subject_num-1,:) = FeatureMap;
            TrainLabel_src(k:k+Train_Subject_num-1,:)  = LabelMap;
            %fprintf("第%d轮训练的训练集：%s\n", j, Train_Subject);
            k = k+Train_Subject_num;
        end
    end
    
    % 统计测试数据
    Test_number = tabulate(TestLabel);
    Train_number = tabulate(TrainLabel_src);        
    % 训练数据处理
    if length(Test_number(:,1)) == 2
        for t = 1 : 3
            if Test_number(1,1) == Train_number(t,1)
                train_0 = Train_number(t,2);
            end
            if Test_number(2,1) == Train_number(t,1)
                train_1 = Train_number(t,2);
            end
        end
        Train_number_min = train_0 + train_1;       
        
        TrainFeature = zeros(Train_number_min, Feature_num);
        TrainLabel = zeros(Train_number_min, 1);     
        
        k = 1;
        for j = 1 : length(TrainLabel_src)
            if TrainLabel_src(j) == Test_number(1,1)
                TrainFeature(k,:) = TrainFeature_src(j,:);
                TrainLabel(k,:) = TrainLabel_src(j,:);
                k = k + 1;
            end
        end
        for j = 1 : length(TrainLabel_src)
            if TrainLabel_src(j) == Test_number(2,1)
                TrainFeature(k,:) = TrainFeature_src(j,:);
                TrainLabel(k,:) = TrainLabel_src(j,:);
                k = k + 1;
            end
        end
    end
    
    % TrainFeature TrainLabel TestFeature TestLabel PredictLabel
    %数据预处理，用matlab自带的mapminmax将训练集和测试集归一化处理[0,1]之间
    %训练数据处理
    [TrainFeature,pstrain] = mapminmax(TrainFeature');
    % 将映射函数的范围参数分别置为0和1
    pstrain.ymin = 0;
    pstrain.ymax = 1;
    % 对训练集进行[0,1]归一化
    [TrainFeature,pstrain] = mapminmax(TrainFeature,pstrain);
    % 测试数据处理
    [TestFeature,pstest] = mapminmax(TestFeature');
    % 将映射函数的范围参数分别置为0和1
    pstest.ymin = 0;
    pstest.ymax = 1;
    % 对测试集进行[0,1]归一化
    [TestFeature,pstest] = mapminmax(TestFeature,pstest);
    % 对训练集和测试集进行转置,以符合libsvm工具箱的数据格式要求
    TrainFeature = TrainFeature';
    TestFeature = TestFeature';

    %寻找最优c和g
    %粗略选择：c&g 的变化范围是 2^(-10),2^(-9),...,2^(10)
    %[bestacc,bestc,bestg] = SVMcg(TrainLabel, TrainFeature,-10,10,-10,10);
    %精细选择：c 的变化范围是 2^(-2),2^(-1.5),...,2^(4), g 的变化范围是 2^(-4),2^(-3.5),...,2^(4)
    %[bestacc,bestc,bestg] = SVMcg(TrainLabel, TrainFeature,-2,4,-4,4,5,0.5,0.5,0.9);

    %训练模型
    %cmd = ['-s ',num2str(0), '-t ',num2str(0), '-c ',num2str(bestc),' -g ',num2str(bestg)];
    cmd = ['-s 0 -t 0 -c 1.2 -g 2.8'];
    model=svmtrain(TrainLabel,TrainFeature,cmd);
    disp(cmd);
    %pause(1);
    %测试分类
    [PredictLabel, accuracy, dec_values]=svmpredict(TestLabel,TestFeature,model);
    
    for p = 1 : length(TestLabel)
        if p == 1
            fprintf("T = ");
        end
        fprintf("%d ", TestLabel(p));
    end
    fprintf("\n"); 
    for p = 1 : length(TestLabel)
        if p == 1
            fprintf("P = ");
        end
        fprintf("%d ", PredictLabel(p));
    end
    fprintf("\n"); 

    % 计算UF1 
    TP_0_i = 0; TP_1_i = 0; TP_2_i = 0;
    FP_0_i = 0; FP_1_i = 0; FP_2_i = 0;
    FN_0_i = 0; FN_1_i = 0; FN_2_i = 0;
    TN_0_i = 0; TN_1_i = 0; TN_2_i = 0;
    for Subject_sample = 1 : Test_num
        % negative：0
        if TestLabel(Subject_sample,1) == 0 && PredictLabel(Subject_sample,1) == 0
            TP_0_i = TP_0_i + 1;
        elseif TestLabel(Subject_sample,1) ~= 0 && PredictLabel(Subject_sample,1) == 0
            FP_0_i = FP_0_i + 1;
        elseif TestLabel(Subject_sample,1) == 0 && PredictLabel(Subject_sample,1) ~= 0
            FN_0_i = FN_0_i + 1;
        elseif TestLabel(Subject_sample,1) == 0 && PredictLabel(Subject_sample,1) == 0
            TN_0_i = TN_0_i + 1;
        end
        % positive：1
        if TestLabel(Subject_sample,1) == 1 && PredictLabel(Subject_sample,1) == 1
            TP_1_i = TP_1_i + 1;
        elseif TestLabel(Subject_sample,1) ~= 1 && PredictLabel(Subject_sample,1) == 1
            FP_1_i = FP_1_i + 1;
        elseif TestLabel(Subject_sample,1) == 1 && PredictLabel(Subject_sample,1) ~= 1
            FN_1_i = FN_1_i + 1;
        elseif TestLabel(Subject_sample,1) == 1 && PredictLabel(Subject_sample,1) == 1
            TN_1_i = TN_1_i + 1;
        end
        % surprise：2
        if TestLabel(Subject_sample,1) == 2 && PredictLabel(Subject_sample,1) == 2
            TP_2_i = TP_2_i + 1;
        elseif TestLabel(Subject_sample,1) ~= 2 && PredictLabel(Subject_sample,1) == 2
            FP_2_i = FP_2_i + 1;
        elseif TestLabel(Subject_sample,1) == 2 && PredictLabel(Subject_sample,1) ~= 2
            FN_2_i = FN_2_i + 1;
        elseif TestLabel(Subject_sample,1) == 2 && PredictLabel(Subject_sample,1) == 2
            TN_2_i = TN_2_i + 1;
        end
    end 
  
    TP_0 = TP_0 + TP_0_i; TP_1 = TP_1 + TP_1_i; TP_2 = TP_2 + TP_2_i;
    FP_0 = FP_0 + FP_0_i; FP_1 = FP_1 + FP_1_i; FP_2 = FP_2 + FP_2_i;
    FN_0 = FN_0 + FN_0_i; FN_1 = FN_1 + FN_1_i; FN_2 = FN_2 + FN_2_i;
    TN_0 = TN_0 + TN_0_i; TN_1 = TN_1 + TN_1_i; TN_2 = TN_2 + FN_2_i;

    Predict(pre:pre+Test_num-1,1) = TestLabel;
    Predict(pre:pre+Test_num-1,2) = PredictLabel;
    pre=pre+Test_num;
    cd('G:\MEGC2019\micro-expression');
end

%%
%打印测试分类结果
% figure;
% hold on;
% plot(Predict(:,1),'o');
% plot(Predict(:,2),'r*');
% legend('实际测试集分类','预测测试集分类');
% title('测试集的实际分类和预测分类图','FontSize',10);
% pause(1);
    
F1_0 = 2 * TP_0 / (2 * TP_0 + FP_0 + FN_0);
F1_1 = 2 * TP_1 / (2 * TP_1 + FP_1 + FN_1);
F1_2 = 2 * TP_2 / (2 * TP_2 + FP_2 + FN_2);
UF1 = (F1_0 + F1_1 + F1_2)/3;
fprintf("UF1=：%d\n", UF1);

UAR = (double(TP_0)./250.0 + TP_1/109.0 + TP_2/83.0)/3;
fprintf("UAR=：%d\n", UAR);
%%
n0_0 = 0; n0_1 = 0; n0_2 = 0;
n1_0 = 0; n1_1 = 0; n1_2 = 0;
n2_0 = 0; n2_1 = 0; n2_2 = 0;
for s = 1 : 145
    % acc
    if Predict(s,1)  == 0 && Predict(s,2) == 0
        n0_0 = n0_0 + 1;
    elseif Predict(s,1)  == 0 && Predict(s,2) == 1
        n0_1 = n0_1 + 1;
    elseif Predict(s,1)  == 0 && Predict(s,2) == 2
        n0_2 = n0_2 + 1;
    elseif Predict(s,1)  == 1 && Predict(s,2) == 0
        n1_0 = n1_0 + 1;
    elseif Predict(s,1)  == 1 && Predict(s,2) == 1
        n1_1 = n1_1 + 1;
    elseif Predict(s,1)  == 1 && Predict(s,2) == 2
        n1_2 = n1_2 + 1;
    elseif Predict(s,1)  == 2 && Predict(s,2) == 0
        n2_0 = n2_0 + 1;
    elseif Predict(s,1)  == 2 && Predict(s,2) == 1
        n2_1 = n2_1 + 1;
    elseif Predict(s,1)  == 2 && Predict(s,2) == 2
        n2_2 = n2_2 + 1;
    end
end
%%
% compute confusion matrix
SUM = n0_0+n0_1+n0_2+n1_0+n1_1+n1_2+n2_0+n2_1+n2_2;
A = [n0_0  n0_1 n0_2; n1_0 n1_1 n1_2; n2_0 n2_1 n2_2];
%%
confusion_matrix(Predict(:,1),Predict(:,2));
%%
n = 0;
for s = 1 : 442
    if Predict(s,1) == Predict(s,2)
        n = n+1;
    end
end
acc = n / 442
%%
FileTabel(:,5) = array2table(Predict(:,2));
FileLabel = table2cell(FileTabel);
fmicro = fopen('micro.txt', 'wt');
k = 1;
for i = 1 : length(Subject_num(:,1))
    fprintf(fmicro,'%s %s\n', cell2mat(FileLabel(k,1)), cell2mat(FileLabel(k,2))); 
    for j = 1 : cell2mat(Subject_num(i,2))
        fprintf(fmicro,'%s %d %d\n', cell2mat(FileLabel(k,3)), cell2mat(FileLabel(k,4)), cell2mat(FileLabel(k,5))); 
        k = k + 1;
    end
end
fclose(fmicro);
%%