tmp_averaged_auc_file = '~/autism/data/gene_scores/on_VAL63.auc'; % Figure 2
% tmp_averaged_auc_file = '~/autism/data/gene_scores/on_PERM63_averaged.auc'; % Figure 3

tmp_output_file = '~/autism/doc/pics/final/aucs_on_VAL63.png'; % Figure 2
% tmp_output_file = '~/autism/doc/pics/final/aucs_on_PERM63.png'; % Figure 3

% load auc data {{{
data = textscan(fopen(tmp_averaged_auc_file, 'r'), '%s%f', 'Delimiter', ' ');

avg_auc_ppi = data{2}(ismember(data{1}, 'ppi'));

% regions
prefix = {'', 'ix', 'ux'};
for i = 1 : 4
  for j = 1 : 3
    tag = sprintf('%spcc-%d-x', prefix{j}, i);
    auc = data{2}(ismember(data{1}, tag));
    if isempty(auc)
      R{i}(j) = NaN;
    else
      R{i}(j) = auc;
    end
  end
end
% periods
for i = 1 : 12
  for j = 1 : 3
    tag = sprintf('%spcc-x-%d', prefix{j}, i + 1);
    auc = data{2}(ismember(data{1}, tag));
    if isempty(auc)
      P{i}(j) = NaN;
    else
      P{i}(j) = auc;
    end
  end
end
% combinations
for i = 1 : 4
  for j = 1 : 12
    for k = 1 : 3
      tag = sprintf('%spcc-%d-%d', prefix{k}, i, j + 1);
      auc = data{2}(ismember(data{1}, tag));
      if isempty(auc)
        C{i, j}(k) = NaN;
      else
        C{i, j}(k) = auc;
      end
    end
  end
end
% }}}

crossplot_auc(tmp_output_file, '', R, P, C, avg_auc_ppi);

clear tmp_* data R P C tag auc i j

% -------------
% Yuxiang Jiang (yuxjiang@indiana.edu)
% Department of Computer Science
% Indiana University, Bloomington
% Last modified: Mon 24 Jul 2021
