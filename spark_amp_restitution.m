% Spark amplitude restitution

clear all

outputs = 200011; % change based on dimensions of the CUDA run
trials = 100;     % change based on dimensions of the CUDA run

%Nopen_all = csvread("N_open.csv");
Cads_all = csvread("Ca_ss.csv");
plottime = csvread("plottime.csv");

%Nopen_adjusted = zeros(outputs, trials);

sparklocs = zeros(trials, 100); %initializing array with extra space
                               % 50 was arbitrarily chosen- change with
                               % longer trials
Cads_amps = zeros(outputs, trials);

% making an adjusted array that ignores quarks
for i=1:outputs
    for j = 1:trials
        if Cads_all(i,j) <= 30
            Cads_amps(i,j) = 0;
        else 
            Cads_amps(i,j) = Cads_all(i,j);
        end
    end
end

for i = 1:trials
    [pks, locs] = findpeaks(Cads_amps(:, i));
    len = length(locs);
    for j= 1:len
        sparklocs(i,j) = locs(j);
    end
end

restitution = zeros(trials, 100);
Cads_amps = Cads_amps.';

for i = 1:trials
    for j = 1:99
        pos = sparklocs(i,j);
        nextspark = sparklocs(i,j+1);
        if nextspark ~= 0
            if pos ~= 0 && Cads_amps(i,nextspark) ~= 0
                restitution(i,j) =  Cads_amps(i,nextspark) /  Cads_amps(i,pos);
            end
        end
    end
end

sparktimes = zeros (trials,100);

for i = 1:trials
    for j = 1:100
        loc = sparklocs(i,j);
        if loc > 0
            time = plottime(loc);
            sparktimes(i,j) = time;
        end
    end
end

sparkdelays = zeros(trials, 100);

for i = 1:trials
    for j = 1:99
        delay = sparktimes(i,j+1)- sparktimes(i,j);
        sparkdelays(i,j) = delay;
    end
end

X = sparkdelays(:);
Y = restitution(:);
idx1 = X > 20;
idx2 = Y > 0;
figure
hold on
scatter(X(idx1), Y(idx2), 'k')
%xlim([0,1500]);
xlabel("Time in ms");
ylabel("A_2/A_1");
title("Spark amplitude restitution");
% p = fit(X(idx1), Y(idx2), 'poly2');
% plot(p, X(idx1), Y(idx2))





        
  
