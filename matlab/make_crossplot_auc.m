% Script
%
% This script makes a cross-sectional pie chart describing averaged AUC over BrainSpan networks and
% with two different approaches to incorporating PPI.
%
% This file assume the precomputed aucs haven't been averaged. For plotting those aucs already
% averaged cases, use make_crossplot_auc_preavg.m
%
% A folder containing cross-validation results.
% One file per network.
% file format (space-splitted)
% <id> <edge cutoff> <propagatoin steps> <auc>
% example: Brain-cv-001 0.75 5 0.7
% Filenames follow <GTEx tissue>.auc, e.g. Brain.auc
tmp_res_dir = '~/autism/data/result/best_network/';

% filename of the output plot
tmp_output_file = '~/autism/doc/pics/final/aucs_on_POS65_cv.png';

% load auc data {{{
avg_auc_ppi = loc_extract_average_auc([tmp_res_dir 'ppi.auc']);
% regions
tmp_prefix = {'', 'ix', 'ux'};
for i = 1 : 4
  for j = 1 : 3
    tmp_filename = sprintf('%s%spcc_%d_x.auc', tmp_res_dir, tmp_prefix{j}, i);
    R{i}(j) = loc_extract_average_auc(tmp_filename);
  end
end
% periods
for i = 1 : 12
  for j = 1 : 3
    tmp_filename = sprintf('%s%spcc_x_%d.auc', tmp_res_dir, tmp_prefix{j}, i + 1);
    P{i}(j) = loc_extract_average_auc(tmp_filename);
  end
end
% combinations
for i = 1 : 4
  for j = 1 : 12
    for k = 1 : 3
      tmp_filename = sprintf('%s%spcc_%d_%d.auc', tmp_res_dir, tmp_prefix{k}, i, j + 1);
      C{i, j}(k) = loc_extract_average_auc(tmp_filename);
    end
  end
end
% }}}

crossplot_auc(tmp_output_file, '', R, P, C, avg_auc_ppi);

clear tmp_* i j R P C

% function loc_extract_average_auc {{{
function [auc, stds] = loc_extract_average_auc(arg)
  if isnumeric(arg)
    auc  = arg;
    stds = 0;
  elseif ischar(arg)
    if ~exist(arg, 'file')
      % msg = sprintf('Cannot open file [%s].', arg);
      % warning('make_crossplot_auc:FileErr', msg);
      auc = NaN;
      stds = NaN;
    else
      % read data
      data = textscan(fopen(arg, 'r'), '%s%s%s%f', 'Delimiter', ' ');
      auc  = mean(data{4});
      stds = std(data{4});
    end
  else
    msg = 'Required either a number or a file name.';
    error('barh_compare_auc3:InputErr', msg);
  end
end
% }}}

% -------------
% Yuxiang Jiang (yuxjiang@indiana.edu)
% Department of Computer Science
% Indiana University, Bloomington
% Last modified: Tue 24 Jul 2021
