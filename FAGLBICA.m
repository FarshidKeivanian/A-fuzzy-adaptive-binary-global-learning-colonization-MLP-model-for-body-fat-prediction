%Hybrid FABEICA-Machine Learning framework
clc;
clear;
close all;

%% Problem Definition
tic;
data = LoadData();

% Probability of applying Velocity Divergence
PVD = 1;

% Probability of applying Velocity Adaptation
PVA = 1;

CostFunction=@(s) FeatureSelectionCost(s,data);        % Cost Function

nVar = data.nx;       % Number of Decision Variables

VarSize = [1 nVar];   % Decision Variables Matrix S

VarMin = -1;            % Lower Bound of Decision Variables
VarMax = 1;            % Upper Bound of Decision Variables

VelMax = 1;
VelMin = -1;

GlobalMin = 0;
%% RK-Fuzzy ICADEPSO Parameters
MaxIt = 100;
nPop = 20;
nEmp = 5;
mu = 0.05;
alpha = 1;            % Selection Pressure
Beta = 1;
pRevolution = 0.1;
zeta = 0.1;           % Colonies Mean Cost Coefficient
nCol = nPop - nEmp;

pDivergence = 0.1;
n = 3;
Pi = 2*(n+1-(1:n))/(n*(n+1));
q = 3;
nFcnEval = inf;
VTR = 1e-4;
w=1;                % Inertia Weight
wdamp=0.99;         % Inertia Weight Damping Ratio
c1=2;               % Global Learning Coefficient
c2=2;               % Personal Learning Coefficient

beta_min=0;   % Lower Bound of Scaling Factor
beta_max=1;   % Upper Bound of Scaling Factor

%% Initialization
empty_country.Position=[];
empty_country.Velocity=[];
empty_country.Cost=[];
empty_country.Out=[];

empty_country.Best.Position=[];
empty_country.Best.Cost=[];
empty_country.Best.Out=[];

country=repmat(empty_country,nPop,1);
%gimp.Cost=inf;

GlobalBest.Cost=inf;

BestCost=zeros(MaxIt,1);

fprintf('Generate initial solutions ... \n');
  for i=1:nPop
      country(i).Position=unifrnd(VarMin,VarMax,VarSize);
      country(i).Velocity=zeros(VarSize);      
      if country(i).Position == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
          f = randi(nVar);
          country(i).Position(f) = 1;
      end
      [country(i).Cost, country(i).Out] =CostFunction(country(i).Position);
      
      country(i).Best.Position = country(i).Position;
      country(i).Best.Cost = country(i).Cost;
      country(i).Best.Out = country(i).Out;
      
      if country(i).Best.Cost < GlobalBest.Cost
              GlobalBest = country(i).Best;
      end                                           
      
  end
  % Sort countries
  [~,index]=sort([country.Cost],'ascend');
  country=country(index);
  % Assign of Colonies and Imperialists
  imp=country(1:nEmp);
  col=country(nEmp+1:end);

    empty_empire.Imp=[];
    empty_empire.Col=repmat(empty_country,0,1);
    empty_empire.nCol=0;
    empty_empire.TotalCost=[];
    emp=repmat(empty_empire,nEmp,1);
        
    % Assign Imperialists
    for k=1:nEmp
        emp(k).Imp=imp(k);
    end
    
    % Assign Colonies
    P1=exp(-alpha*[imp.Cost]/max([imp.Cost]));
%    P=P/sum(P);
    Sum_P1 = 0;
    for bbb = 1:numel(emp)
    Sum_P1 = P1(1,bbb) + Sum_P1;
    end
    NP1 = P1/Sum_P1;
    
    for Num=1:nCol
        
        k=RouletteWheel(NP1);
        
        emp(k,1).Col=[emp(k,1).Col
                    col(Num)];
        
        emp(k,1).nCol=emp(k,1).nCol+1;
    end
fprintf('Main Loop ... \n');
Costs = zeros (MaxIt, 1);

 %% Main Loop
for it=1:MaxIt
   % Velocity Adaptation
    nEmp=numel(emp);
    if rand < PVA
    for k=1:nEmp
        if size(emp(k,1).Col,2) ~= 1
             emp(k,1).Col = emp(k,1).Col';
        end
        for col=1:numel(emp(k).Col)
                if numel(emp(k).Col) ~= 0                                    
            d=emp(k).Imp.Position-emp(k).Col(col).Position;
            d2=emp(k).Col(col).Best.Position-emp(k).Col(col).Position;
            d3=GlobalBest.Position-emp(k).Col(col).Position;   
            NCost = abs((emp(k).Col(col).Cost-emp(k).Imp.Cost)/emp(k).Col(col).Cost);            
            NIt = abs(it/MaxIt);
            UFuzzy = [NCost; NIt];
            FISMAT = readfis('FIS13.fis');
            Y = evalfis(UFuzzy, FISMAT);
            w = Y(1,1);
            Beta = Y(1,2);
            c1 = Y(1,3);
            c2 = Y(1,4);
            pDivergence = Y(1,5);
            F1 = Y(1,6);
            pCR = Y(1,7);   
            if it==10 || it==20 || it==30 || it==40 || it==50 || it==60 || it==70 || it==80 || it==90 || it==100
                ConvergenceRate=((BestCost(it-9) - GlobalBest.Cost))/10;  % Time Window = 10              
                % GlobalBest.Cost = BestCost(it-1);
                I = [GlobalBest.Cost - GlobalMin; BestCost(it-2)- GlobalMin;BestCost(it-3)- GlobalMin;BestCost(it-4)- GlobalMin;BestCost(it-5)- GlobalMin;BestCost(it-6)- GlobalMin;BestCost(it-7)- GlobalMin;BestCost(it-8)- GlobalMin;BestCost(it-9)- GlobalMin];
                Accuracy = 1-((GlobalBest.Cost - GlobalMin)/sum(I));
                UFuzzy =[ConvergenceRate; Accuracy];            
                FISMAT = readfis('FIS14.fis');
                Y = evalfis(UFuzzy, FISMAT);
                PVD = Y(1,1);
                PVA = Y(1,2);
            end
            r1 = VelMin + (2*VelMax)*rand(VarSize);
            r2 = VelMin + (2*VelMax)*rand(VarSize);
            r3 = VelMin + (2*VelMax)*rand(VarSize);            
            x = w.*emp(k).Col(col).Velocity + Beta.*r1.*d + c1.*r2.*d2 + c2.*r3.*d3;
            x=min(max(x,VelMin),VelMax);           
            xnew = emp(k).Col(col).Position + x;                        
            emp(k).Col(col).Position = xnew;  
            flag=(emp(k).Col(col).Position<VarMin | emp(k).Col(col).Position>VarMax);
            emp(k).Col(col).Velocity(flag) = -emp(k).Col(col).Velocity(flag);              
            emp(k).Col(col).Position=min(max(emp(k).Col(col).Position,VarMin),VarMax);                        
            if emp(k).Col(col).Position == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
               f = randi(nVar);
               emp(k).Col(col).Position(f) = 1;
            end          
            [emp(k).Col(col).Cost, emp(k).Col(col).Out] = CostFunction(emp(k).Col(col).Position);
            if emp(k).Col(col).Cost < emp(k).Imp.Cost
                a=emp(k).Col(col);
                b=emp(k).Imp;
                emp(k).Imp = a;
                emp(k).Col(col) = b;
            end           
                end
        end
            d1=emp(k).Imp.Best.Position-emp(k).Imp.Position;
            d2=GlobalBest.Position-emp(k).Imp.Position;       
            r1 = VelMin + (2*VelMax)*rand(VarSize);
            r2 = VelMin + (2*VelMax)*rand(VarSize);           
            x = w.*emp(k).Imp.Velocity + c1.*r1.*d1 + c2.*r2.*d2;                                  
            x=min(max(x,VelMin),VelMax);            
            xnew = emp(k).Imp.Position + x;
            emp(k).Imp.Position = xnew;            
            flag=(emp(k).Imp.Position<VarMin | emp(k).Imp.Position>VarMax);
            emp(k).Imp.Velocity(flag) = -emp(k).Imp.Velocity(flag);   
            emp(k).Imp.Position=min(max(emp(k).Imp.Position,VarMin),VarMax);                        
            if emp(k).Imp.Position == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                f = randi(nVar);
                emp(k).Imp.Position(f) = 1;
            end             
            [emp(k).Imp.Cost, emp(k).Imp.Out] = CostFunction(emp(k).Imp.Position);    
    end
    end
% Velocity Divergence
if rand < PVD
    sigma=0.1*(VarMax-VarMin);
    nmu=ceil(mu*nVar);
    nEmp=numel(emp);
for k=1:nEmp  % Az har Emperatori (Imperialist) ke 150 color data darad, 8 ta az 150 color data ra tagheer midahim
    if rand<pDivergence
        NewVel = emp(k).Imp.Velocity + sigma*randn(VarSize);
        emp(k).Imp.Velocity=min(max(NewVel,VelMin),VelMax);
        xnew = emp(k).Imp.Position + emp(k).Imp.Velocity;        
        emp(k).Imp.Position = xnew;        
        flag=(emp(k).Imp.Position<VarMin | emp(k).Imp.Position>VarMax);
        emp(k).Imp.Velocity(flag) = -emp(k).Imp.Velocity(flag);                      
        emp(k).Imp.Position=min(max(emp(k).Imp.Position,VarMin),VarMax);                        
        if emp(k).Imp.Position == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
               f = randi(nVar);
               emp(k).Imp.Position(f) = 1;
        end                              
        [emp(k).Imp.Cost, emp(k).Imp.Out]=CostFunction(emp(k).Imp.Position);                
    end
        for i=1:numel(emp(k).Col)
            if rand<=pDivergence
                
                NewVel = emp(k).Col(i).Velocity + sigma*randn(VarSize);
                emp(k).Col(i).Velocity=min(max(NewVel,VelMin),VelMax);
                xnew = emp(k).Col(i).Position + emp(k).Col(i).Velocity;
                emp(k).Col(i).Position = xnew;                
                flag=(emp(k).Col(i).Position<VarMin | emp(k).Col(i).Position>VarMax);
                emp(k).Col(i).Velocity(flag) = -emp(k).Col(i).Velocity(flag);              
                emp(k).Col(i).Position=min(max(emp(k).Col(i).Position,VarMin),VarMax);                        
                 if emp(k).Col(i).Position == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    f = randi(nVar);
                    emp(k).Col(i).Position(f) = 1;
                 end                   
                [emp(k).Col(i).Cost, emp(k).Col(i).Out]=CostFunction(emp(k).Col(i).Position);                                 
            end
        end 
end
end
%% Intra-Empire Competition
    nEmp=numel(emp);
    for k=1:nEmp
        for i=1:emp(k).nCol
            if emp(k).Col(i).Cost<emp(k).Imp.Cost
                imp=emp(k).Imp;
                col=emp(k).Col(i);
                
                emp(k).Imp=col;
                emp(k).Col(i)=imp;
            end
        end
    end

%% Total Cost
nEmp=numel(emp);

for k=1:nEmp
    if emp(k).nCol>0
        emp(k).TotalCost=emp(k).Imp.Cost+zeta*mean([emp(k).Col.Cost]);
    else
        emp(k).TotalCost=emp(k).Imp.Cost;
    end
end

% Inter Empire Competition
nEmp = numel(emp);
if nEmp > 1
TotalCost=[emp.TotalCost];

[~, WeakestEmpIndex]=max(TotalCost);
WeakestEmp=emp(WeakestEmpIndex);

P=exp(-alpha*TotalCost/max(TotalCost));
P(WeakestEmpIndex)=0;
P=P/sum(P);
if any(isnan(P))
    P(isnan(P))=0;
    if all(P==0)
        P(:)=1;
    end
    P=P/sum(P);
end

if WeakestEmp.nCol>0
    [~, WeakestColIndex]=max([WeakestEmp.Col.Cost]);
    WeakestCol=WeakestEmp.Col(WeakestColIndex);

    WinnerEmpIndex=RouletteWheelSelection(P);
    WinnerEmp=emp(WinnerEmpIndex);

    WinnerEmp.Col(end+1)=WeakestCol;
    WinnerEmp.nCol=WinnerEmp.nCol+1;
    emp(WinnerEmpIndex)=WinnerEmp;

    WeakestEmp.Col(WeakestColIndex)=[];
    WeakestEmp.nCol=WeakestEmp.nCol-1;
    emp(WeakestEmpIndex)=WeakestEmp;
end

if WeakestEmp.nCol==0

    WinnerEmpIndex2=RouletteWheelSelection(P);
    WinnerEmp2=emp(WinnerEmpIndex2);

    WinnerEmp2.Col(end+1)=WeakestEmp.Imp;
    WinnerEmp2.nCol=WinnerEmp2.nCol+1;
    emp(WinnerEmpIndex2)=WinnerEmp2;

    emp(WeakestEmpIndex)=[];
end

end
 % Updates and then Competition
   for i=1:numel(emp)
       for j=1:emp(i).nCol
                  if emp(i).Col(j).Cost < emp(i).Col(j).Best.Cost            
                    emp(i).Col(j).Best.Position = emp(i).Col(j).Position;
                    emp(i).Col(j).Best.Cost = emp(i).Col(j).Cost;
                    emp(i).Col(j).Best.Out = emp(i).Col(j).Out;            
                        if emp(i).Col(j).Best.Cost < GlobalBest.Cost
                           GlobalBest = emp(i).Col(j).Best;
                       end                                 
                  end        
       end       
         if emp(i).Imp.Cost < emp(i).Imp.Best.Cost            
            emp(i).Imp.Best.Position = emp(i).Imp.Position;
            emp(i).Imp.Best.Cost = emp(i).Imp.Cost;
            emp(i).Imp.Best.Out = emp(i).Imp.Out;            
               if emp(i).Imp.Best.Cost < GlobalBest.Cost
                  GlobalBest = emp(i).Imp.Best;
               end                                 
          end          
   end   
BestCost(it) = GlobalBest.Cost;

% Update Best Solution Ever Found
%     imp=[emp.Imp];
%     [~, BestImpIndex]=min([imp.Cost]);
%     GlobalBest = emp(BestImpIndex).Imp;
%     BestCost(it) = emp(BestImpIndex).Imp.Cost;   
end

%% Results
time = toc;
time = time/100; % for each iteration time (100 iteration time)
time = time/5; % for each learning time (5 times nRun for learning)
save('BestCost.mat', 'BestCost');
save('GlobalBest.mat', 'GlobalBest');
save('time.mat', 'time');
