% Output figure folder
pic_dir = '../doc/pics/calibrated/';

log_scale = false;
no_seed = false;
tail = 0.95;

% gene score directory
% gs_dir is the directory where gene score stores ([0, 1])
% These scores are used for pos65, krishnan, duda and all_ones (MutPred)
gs_dir = '../data/gene_scores/calibrated/';


% gs_dir is the directory where log-scaled gene score stores
% These scores are used for all the rest of gene scores, set it as the same as gs_dir if not using
% log-scaled scores
% gsLog_dir = '~/autism/data/gene_scores/calibrated_log/';
gsLog_dir = '../data/gene_scores/calibrated/';

% ======== DO NOT MODIFY BELOW THIS LINE ========

% [0, 1]-scaled calibrated MutPred scores
tmp_mpFile = '../data/mutpred.txt';
if log_scale
  tmp_mpLogFile = '../data/mutpred_log.txt';
else
  tmp_mpLogFile = tmp_mpFile;
end

tmp_prog = '../cpp/bin/tail-stat.run';
% vartype:
% '': all
% 'missense_'
% 'lof_'
% 'indel_'
tmp_vartypes = {'', 'missense_', 'lof_', 'indel_'};

% category: regions, periods, combinations
tmp_categories = {'regions', 'periods', 'combinations'};

% seed gene list
tmp_sdFile = '../data/gene-lists/POS65.entrez';
tmp_colors = pfp_cbrewer(8, 'light');
tmp_colors = 1 - (1 - tmp_colors) / 2; % lighter

network_prefix = 'pcc_';
ixmerge_prefix = 'ix';
uxmerge_prefix = 'ux';

for ii = 1 : numel(tmp_vartypes)
  tmp_vartype = tmp_vartypes{ii};

  if isempty(tmp_vartype)
    printable_type = 'all';
  else
    printable_type = tmp_vartype(1:end-1);
  end
  fprintf('%s\n', printable_type);

  tmp_csFile = ['~/autism/data/denovo/lists/' tmp_vartype 'cases.txt'];
  tmp_cnFile = ['~/autism/data/denovo/lists/' tmp_vartype 'controls.txt'];
  if no_seed
    tmp_cmdPfx = strjoin(...
      {tmp_prog, '-c', tmp_csFile, '-t', tmp_cnFile, '-v', tmp_mpFile, ...
      '-q', sprintf('%d', tail), '-excl', tmp_sdFile}, ' ');
  else
    tmp_cmdPfx = strjoin(...
      {tmp_prog, '-c', tmp_csFile, '-t', tmp_cnFile, '-v', tmp_mpFile, ...
      '-q', sprintf('%d', tail)}, ' ');
  end

  if log_scale
    if no_seed
      tmp_cmdLogPfx = strjoin(...
        {tmp_prog, '-c', tmp_csFile, '-t', tmp_cnFile, '-v', tmp_mpLogFile, ...
        '-q', sprintf('%d', tail), '-excl', tmp_sdFile, '-log y'}, ' ');
    else
      tmp_cmdLogPfx = strjoin(...
        {tmp_prog, '-c', tmp_csFile, '-t', tmp_cnFile, '-v', tmp_mpLogFile, ...
        '-q', sprintf('%d', tail), '-log y'}, ' ');
    end
  else
    tmp_cmdLogPfx = tmp_cmdPfx;
  end

  for jj = 1 : numel(tmp_categories)
    tmp_category = tmp_categories{jj};
    % one time P-values {{{
    % specials {{{
    lcs_specials = {...
      'MutPred',  tmp_colors(1, :); ...
      'POS65',    tmp_colors(4, :); ...
      'Krishnan', tmp_colors(5, :); ...
      'Duda',     tmp_colors(8, :); ...
      'PPI',      [0 0 0];
    };

    % tmp_names must be one-by-one mapping to lcs_specials!
    tmp_names = {...
      'all_ones.out', ...
      'pos65.out', ...
      'krishnan.out', ...
      'duda.out', ...
      'ppi.out'...
      };
    pvalues_specials = cell(1, numel(tmp_names));
    for i = 1 : numel(tmp_names)
      if strcmp(tmp_names{i}, 'ppi.out')
        file = [gsLog_dir tmp_names{i}];
        cmd = strjoin({tmp_cmdLogPfx, '-g', file}, ' ');
        [~, tmp_pvalue] = system(cmd);
      else
        file = [gs_dir tmp_names{i}];
        cmd = strjoin({tmp_cmdPfx, '-g', file}, ' ');
        [~, tmp_pvalue] = system(cmd);
      end
      pvalues_specials{i} = str2double(tmp_pvalue);
    end
    % }}}

    if strcmp(tmp_category, 'regions')
      % regions {{{
      fprintf('calculating p-values for regions\n');

      regions = {'1_x', '2_x', '3_x', '4_x'};
      lcs_regions  = cell(numel(regions), 2);
      pvalues_regions = cell(1, numel(regions));

      for i = 1 : numel(regions)
        % label and color
        lcs_regions{i, 1} = ['region ' regexprep(regions{i}, '([0-9])_x', '$1')];
        lcs_regions{i, 2} = tmp_colors(2, :);

        % single
        file = [gsLog_dir network_prefix regions{i} '.out'];
        [~, tmp_pvalue] = system(strjoin({tmp_cmdLogPfx, '-g', file}, ' '));
        pvalues_regions{i}{1} = str2double(tmp_pvalue);

        % i-merged
        file = [gsLog_dir ixmerge_prefix network_prefix regions{i} '.out'];
        [~, tmp_pvalue] = system(strjoin({tmp_cmdLogPfx, '-g', file}, ' '));
        pvalues_regions{i}{2} = str2double(tmp_pvalue);

        % u-merged
        file = [gsLog_dir uxmerge_prefix network_prefix regions{i} '.out'];
        [~, tmp_pvalue] = system(strjoin({tmp_cmdLogPfx, '-g', file}, ' '));
        pvalues_regions{i}{3} = str2double(tmp_pvalue);
      end
      % }}}
    elseif strcmp(tmp_category, 'periods')
      % periods {{{
      fprintf('calculating p-values for periods\n');

      periods = {...
        'x_2', 'x_3', 'x_4',  'x_5',  'x_6',  'x_7', ...
        'x_8', 'x_9', 'x_10', 'x_11', 'x_12', 'x_13', 'prenatal', 'postnatal'};
      lcs_periods  = cell(numel(periods), 2);
      pvalues_periods = cell(1, numel(periods));

      for i = 1 : numel(periods)
        % label and color
        lcs_periods{i, 1} = ['period ' regexprep(periods{i}, 'x_([0-9]+)', '$1')];
        lcs_periods{i, 2} = tmp_colors(3, :);

        % single
        file = [gsLog_dir network_prefix periods{i} '.out'];
        [~, tmp_pvalue] = system(strjoin({tmp_cmdLogPfx, '-g', file}, ' '));
        pvalues_periods{i}{1} = str2double(tmp_pvalue);

        % i-merged
        file = [gsLog_dir ixmerge_prefix network_prefix periods{i} '.out'];
        [~, tmp_pvalue] = system(strjoin({tmp_cmdLogPfx, '-g', file}, ' '));
        pvalues_periods{i}{2} = str2double(tmp_pvalue);

        % u-merged
        file = [gsLog_dir uxmerge_prefix network_prefix periods{i} '.out'];
        [~, tmp_pvalue] = system(strjoin({tmp_cmdLogPfx, '-g', file}, ' '));
        pvalues_periods{i}{3} = str2double(tmp_pvalue);
      end
      % }}}
    elseif strcmp(tmp_category, 'combinations')
      % combinations {{{
      fprintf('calculating p-values for combinations\n');

      combs = {...
        '1_3', '1_4',  '1_5',  '1_6', '1_7', '1_8', '1_10', '1_11', '1_12', '1_13', ...
        '2_2', '2_3',  '2_4',  '2_5', '2_6', '2_7', '2_8',  '2_10', '2_11', '2_12', '2_13', ...
        '3_2', '3_3',  '3_4',  '3_5', '3_7', '3_8', '3_10', '3_11', '3_12', '3_13', ...
        '4_5', '4_10', '4_12', '4_13'};
      lcs_combs  = cell(numel(combs), 2);
      pvalues_combs = cell(1, numel(combs));

      for i = 1 : numel(combs)
        % label and color
        region_index = ['R' regexprep(combs{i}, '([0-9])_.*', '$1')];
        period_index = ['P' regexprep(combs{i}, '.*_([0-9]+)', '$1')];
        lcs_combs{i, 1} = sprintf('%s,%3s', region_index, period_index);
        lcs_combs{i, 2} = tmp_colors(6, :);

        % single
        file = [gsLog_dir network_prefix combs{i} '.out'];
        [~, tmp_pvalue] = system(strjoin({tmp_cmdLogPfx, '-g', file}, ' '));
        pvalues_combs{i}{1} = str2double(tmp_pvalue);

        % i-merged
        file = [gsLog_dir ixmerge_prefix network_prefix combs{i} '.out'];
        [~, tmp_pvalue] = system(strjoin({tmp_cmdLogPfx, '-g', file}, ' '));
        pvalues_combs{i}{2} = str2double(tmp_pvalue);

        % u-merged
        file = [gsLog_dir uxmerge_prefix network_prefix combs{i} '.out'];
        [~, tmp_pvalue] = system(strjoin({tmp_cmdLogPfx, '-g', file}, ' '));
        pvalues_combs{i}{3} = str2double(tmp_pvalue);
      end
      % }}}
    else
      % do nothing
    end
    % }}}

    % plotting parameter {{{
    if strcmp(tmp_vartype, 'missense_')
      pRange = 8;
      labeled = true;
      ptitle = 'Missense';
      fWidth = 3.5;
    elseif strcmp(tmp_vartype, 'lof_')
      pRange = 16;
      labeled = false;
      ptitle = 'Loss-of-function';
      fWidth = 4;
    elseif strcmp(tmp_vartype, 'indel_')
      pRange = 3;
      labeled = false;
      ptitle = 'Indel';
      fWidth = 3;
    else % 'all
      pRange = 20;
      labeled = true;
      ptitle = 'All variant types';
      fWidth = 4;
    end
    % }}}

    % combine plotting data {{{
    if strcmp(tmp_category, 'regions')
      lcs     = [lcs_specials; lcs_regions];
      pvalues = [pvalues_specials, pvalues_regions];
    elseif strcmp(tmp_category, 'periods')
      lcs     = [lcs_specials; lcs_periods];
      pvalues = [pvalues_specials, pvalues_periods];
    elseif strcmp(tmp_category, 'combinations')
      lcs     = [lcs_specials; lcs_combs];
      pvalues = [pvalues_specials, pvalues_combs];
    else
      % do nothing
    end
    % }}}

    barh_compare_pvalue3(...
      [pic_dir tmp_vartype tmp_category '_pvalues.png'], ...
      ptitle, ...
      lcs, pvalues, ...
      'labeled', labeled, 'range', pRange, 'fwidth', fWidth ...
      );
  end
end

clear tmp_* *_prefix
