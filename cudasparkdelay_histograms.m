% An identifiable spark is defined by 5 or more open ryanodine channels



clear all

outputs = 25011; % change based on dimensions of the CUDA run
trials = 100;     % change based on dimensions of the CUDA run

Nopen_all = csvread("N_open.csv");
plottime = csvread("plottime.csv");

Nopen_adjusted = zeros(outputs, trials);

sparklocs = zeros(trials, 50); %initializing array with extra space
                               % 50 was arbitrarily chosen- change with
                               % longer trials

% making an adjusted array that ignores quarks
for i=1:outputs
    for j = 1:trials
        if Nopen_all(i,j) <= 5
            Nopen_adjusted(i,j) = 0;
        else 
            Nopen_adjusted(i,j) = Nopen_all(i,j);
        end
    end
end

for i = 1:trials
    [pks, locs] = findpeaks(Nopen_adjusted(:, i));
    len = length(locs);
    for j= 1:len
        sparklocs(i,j) = locs(j);
    end
end

sparktimes = zeros (trials,50);

for i = 1:trials
    for j = 1:50
        loc = sparklocs(i,j);
        if loc > 0
            time = plottime(loc);
            sparktimes(i,j) = time;
        else 
            sparktimes(i,j) = 0;
        end
    end
end

sparkdelays = zeros(trials, 50);

for i = 1:trials
    for j = 1:49
        delay = sparktimes(i,j+1)- sparktimes(i,j);
        sparkdelays(i,j) = delay;
    end
end

idx = sparkdelays > 20; % ignoring peaks found by findpeaks() that are in the same spark

mean = round(mean(sparkdelays(idx)));
median = round(median(sparkdelays(idx)));



figure
hold on
histogram(sparkdelays(idx), 'Normalization', 'probability')
mnlabel=sprintf('Mean: %3.2d', mean);
mdlabel=sprintf('Median: %3.2d', median);
text=annotation('textbox',[0.58 0.75 0.1 0.1]);
set(text,'String',{mnlabel, mdlabel});
xlabel("Time in ms");
ylabel("Probability");
title("Spark to Spark Delays");
