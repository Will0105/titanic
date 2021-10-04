clear
clc
%%
%导入泰坦尼克号数据

Train = readtable('train.csv');
Test = readtable('test.csv');
%%
%根据船票等级补全各船舱等级下缺失的年龄和票价数据
model = fitcnb(Train,'Survived');
label=predict(model,Test);
%统计年龄船票关于船舱等级的数据
Surv_Pclass = grpstats(Train(:,{'Survived','Pclass','Age','Fare'}),'Pclass');
%头等舱、二等舱、三等舱乘客平均年龄
for i = 1 : height(Surv_Pclass)
    avgAge(i) = Surv_Pclass.mean_Age(i);
end
%补全各船舱乘客的缺失年龄数据
Pclassnum = height(Surv_Pclass);
for i = 1 : Pclassnum
    %figure(1);imagesc(ismissing(Train))
    Train.Age(isnan(Train.Age) & Train.Pclass == i) = avgAge(i);
    Test.Age(isnan(Test.Age) & Test.Pclass == i) = avgAge(i);
end
%%
%计算先验概率与条件概率

%划分年龄区间
Agelevel = 10;
Agenum = max(ceil(Test.Age/Agelevel));
Train.Age = ceil(Train.Age/Agelevel)*Agelevel;
Test.Age = ceil(Test.Age/Agelevel)*Agelevel;
%做存活率关于性别、船票等级、年龄三个特征的统计
Surv_Sex = grpstats(Train(:,{'Survived','Sex'}),{'Survived','Sex'});
Surv_Pclass = grpstats(Train(:,{'Survived','Pclass'}),{'Survived','Pclass'});
Surv_Age = grpstats(Train(:,{'Survived','Age'}),{'Survived','Age'});
%计算先验概率与条件概率
for i = [1,2]
    %拉普拉斯平滑
    P_Survive(i) = (sum(Train.Survived == i-1)+1)/(length(Train.Survived)+2);
    for j = 1 : Agenum
        if ismember(j * Agelevel,Surv_Age.Age(Surv_Age.Survived == i-1))
            P_Age(i,j) = (Surv_Age.GroupCount(Surv_Age.Survived == i-1 & Surv_Age.Age == j * Agelevel)+1)/(sum(Surv_Age.GroupCount(Surv_Age.Survived == i-1))+Agenum);
        else
            P_Age(i,j)=1/(sum(Surv_Age.GroupCount(Surv_Age.Survived == i-1))+Agenum);
        end
    end
    P_Sex(i,:) = (Surv_Sex.GroupCount(Surv_Sex.Survived == i-1)+1)/(sum(Surv_Sex.GroupCount(Surv_Sex.Survived == i-1))+2);
    P_Pclass(i,:) = (Surv_Pclass.GroupCount(Surv_Pclass.Survived==i-1)+1)/(sum(Surv_Pclass.GroupCount(Surv_Pclass.Survived==i-1))+3);
end
%%

%计算后验概率
isfemale = ismember(Test.Sex(:),'female');
P_Alive_Sex = P_Sex(2,2)*isfemale+P_Sex(2,1)*(1-isfemale);
P_Dead_Sex = P_Sex(1,2)*isfemale+P_Sex(1,1)*(1-isfemale);
P_Alive_Age = 0;
P_Dead_Age = 0;
for i = 1 : Agenum
    P_Alive_Age = P_Alive_Age+P_Age(2,i)*(Test.Age == i * Agelevel);
    P_Dead_Age = P_Dead_Age+P_Age(1,i)*(Test.Age == i * Agelevel);
end
P_Alive_Pclass = 0;
P_Dead_Pclass = 0;
for i = 1 : Pclassnum
    P_Alive_Pclass = P_Alive_Pclass + P_Pclass(2,i)*(Test.Pclass == i );
    P_Dead_Pclass = P_Dead_Pclass + P_Pclass(1,i)*(Test.Pclass == i );
end
P_Alive(1,:)=P_Survive(2).*P_Alive_Sex.*P_Alive_Age.*P_Alive_Pclass;
P_Dead(1,:)=P_Survive(1).*P_Dead_Sex.*P_Dead_Age.*P_Dead_Pclass;
Survive=P_Alive-P_Dead;
Final(:,1)=892:1:1309;
Final(:,2)=Survive(1,:)>0;
title = {'PassengerId','Survived'};
result_table = table(Final(:,1),Final(:,2),'VariableNames',title);
writetable(result_table,'submission.csv');
