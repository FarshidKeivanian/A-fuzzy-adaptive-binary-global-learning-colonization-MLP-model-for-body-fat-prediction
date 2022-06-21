figure;
time = load('time.mat'); time = time.time;
BestCost = load('BestCost.mat'); BestCost = BestCost.BestCost;
GlobalBest = load('GlobalBest.mat'); GlobalBest = GlobalBest.GlobalBest;
plot(BestCost,'LineWidth',2);
xlabel('Iteration');
ylabel('Best Cost');
grid on;
hold on;