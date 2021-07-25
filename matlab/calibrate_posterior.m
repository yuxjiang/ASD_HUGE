function [] = calibrate_posterior(ifile, ofile, c, prior, opt)
  %CALIBRATE_POSTERIOR
  %
  % [] = CALIBRATE_POSTERIOR(ifile, ofile, c, prior);
  % [] = CALIBRATE_POSTERIOR(ifile, ofile, c, prior, opt);
  %
  %   Calibrate scores to be posterior probabilities.
  %
  % Input
  % -----
  % (required)
  % [char]
  % ifile:  The input file, raw prediction scores. Format: two-column TSV.
  %         <id> <score>
  %
  % [char]
  % ofile:  The output file, calibrated scores.
  %
  % [double]
  % c:      The proportion of |positive|/|unlabeled| in the training set that
  %         produces the raw prediction. E.g. c = 1 for a balanced training set.
  %         It must be positive.
  %
  % [double]
  % prior:  The estimated prior from running AlphaMax algorightm.
  %         It must be within the interval (0, 1).
  %
  % (optional)
  % [double]
  % opt:    (Tweak) The choice of mapping function to [0, 1]
  %         0: (no op) leave the score as is
  %         1: (clamping) hard clampping [0, 1]
  %         2: (0-1 normalization)
  %         3: output log scaled probabilities
  %         default: 1
  %
  % Output
  % ------
  % None.

  % check inputs {{{
  if nargin ~= 4 && nargin ~= 5
    error('calibrate_posterior:InputCount', 'Expected 4 or 5 inputs.');
  end

  if nargin == 4
    opt = 1;
  end

  % ifile
  validateattributes(ifile, {'char'}, {'nonempty'}, '', 'ifile', 1);
  fin = fopen(ifile, 'r');
  if fin == -1
    error('calibrate_posterior:FileErr', 'Cannot open the input file.');
  end

  % ofile
  validateattributes(ofile, {'char'}, {'nonempty'}, '', 'ofile', 2);
  fout = fopen(ofile, 'w');
  if fout == -1
    error('calibrate_posterior:FileErr', 'Cannot open the output file.');
  end

  % c
  validateattributes(c, {'double'}, {'positive'}, '', 'c', 3);

  % prior
  validateattributes(prior, {'double'}, {'>', 0, '<', 1}, '', 'prior', 4);

  % opt
  validateattributes(opt, {'double'}, {'integer'}, '', 'opt', 5);
  % }}}

  % read raw scores {{{
  data = textscan(fin, '%s%f', 'delimiter', '\t', 'EmptyValue', 0, 'TreatAsEmpty', {'NAS', 'STL'});
  fclose(fin);
  validateattributes(data{2}, {'double'}, {'>=', 0, '<=', 1}, '', 'ifile', 1);
  d = data{2};
  % }}}

  % setting {{{
  % c = 1; % since it was trained on a balanced set
  % prior = 0.015; % pre-estimated class prior
  % }}}

  % estimate and write output {{{
  tol = 1e-8;
  set_as_one = find((1 - d) < tol);
  calibr     = find((1 - d) >= tol);

  p = c .* prior .* d(calibr) ./ (1 - d(calibr));

  switch opt
    case 1
      % clamp
      p = min(1, max(0, p));
    case 2
      % normalization by excluding right-tail outliers
      sigma = std(p);
      mu    = mean(p);
      % fprintf('sigma = %f\n', sigma);
      % fprintf('   mu = %f\n', mu);
      % fprintf('  max = %f\n', max(p));
      p = (p - min(p)) / (3 * sigma);
      p = min(1, max(0, p));
    otherwise % opt == 0 or else
      % no op
  end

  % output
  d(calibr)     = p;
  d(set_as_one) = 1;

  if opt == 3
    d = log(d);
  end

  for i = 1 : numel(data{1})
    fprintf(fout, '%s\t%f\n', data{1}{i}, d(i));
  end
  fclose(fout);
  % }}}
end

% -------------
% Yuxiang Jiang (yuxjiang@indiana.edu)
% Department of Computer Science
% Indiana University, Bloomington
% Last modified: Mon 25 Jan 2021 01:08:17 AM E
