#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cstdlib>

#include <libcb/util.h>
#include <libcb/evaluation.h>
#include <libcb/distribution.h>

using namespace std;

#define ERRMSG(x) error_and_exit(__PRETTY_FUNCTION__, x)

enum Stat {
  PVALUE, //!< p-value.
  AUC,    //!< Area under the ROC curve.
  DOR,    //!< Diagnostic Likelihood Ratio
  LR      //!< Likelihood Ratio
};

struct Options {
  string cfile; // case SNP file.
  string tfile; // control SNP file.
  string vfile; // variant score file.
  string gfile; // gene score file.
  string efile = ""; // exclude gene file.
  string sfile = ""; // seed gene file. (whose score should be all one)
  string mfile = ""; // mild mutation file.
  bool log = false; // whether the scores are logarithms
  Stat statistic = PVALUE; // the choice of statistic
  double vcutoff = 0.0; // the deleterious cutoff.
  double q       = 0.95; // tail determining quantile.
  int outfmt = 0;
  bool bootstrap = false;
} opts;

void usage() {
  cerr
    << "NAME\n"
    << "    tail-stat.run - tail statistics (case vs. control)\n"
    << "SYNOPSIS\n"
    << "\n"
    << "    tail-stat.run -c <CASE FILE> -t <CONTROL FILE> -v <VARIANT FILE> -g <GENE FILE> [OPTION]...\n"
    << "\n"
    << "DESCRIPTION\n"
    << "    This program returns the intended tail statistics.\n"
    << "\n"
    << "MANDATORY ARGUMRNTS\n"
    << "    -c <CASE FILE>\n"
    << "        A parsed SNP file from the case group, with format:\n"
    << "        <Entrez ID>_<Mutation identifier>\n"
    << "        where <Mutation identifier> could be any continuous string.\n"
    << "\n"
    << "    -t <CONTROL FILE>\n"
    << "        A parsed SNP file from the control group, with format:\n"
    << "        <Entrez ID>_<Mutation identifier>\n"
    << "\n"
    << "    -v <VARIANT FILE>\n"
    << "        MutPred score for each SNP, with two-column CSV, delimited by SPACE/TAB\n"
    << "        <Entrez ID>_<Mutation identifier>\n"
    << "\n"
    << "    -g <GENE FILE>\n"
    << "        Gene scores, two-column CSV, delimited by SPACE/TAB\n"
    << "        <Entrez ID> <score>\n"
    << "\n"
    << "OPTIONAL ARGUMENTS\n"
    << "    -vcutoff <Real, (0, 1)> default: 0\n"
    << "        A \"deleterious\" cutoff for mutations scores in probability.\n"
    << "    -vmild <FILE> default: (EMPTY)\n"
    << "        A file of \"mild\" mutations.\n"
    << "        Scores on these mild mutations are set to zero.\n"
    << "    -excl <GENE LIST FILE> default: "" (EMPTY)\n"
    << "        A file of genes whose score is excluded.\n"
    << "    -enforce <GENE LIST FILE> default: "" (EMPTY)\n"
    << "        A file of genes whose score is enforced to be one (or zero if log).\n"
    << "    -stat <pvalue|auc|dor|lr> default: pvalue\n"
    << "        The choice of statistics.\n"
    << "    -q <Real, (0, 1)> default: .95\n"
    << "        The quantile to determine TAIL\n"
    << "    -log <y|n> default: n\n"
    << "        Whether to treat the input gene and variant scores as log prob.\n"
    << "    -outfmt <0|1|2> default: 0\n"
    << "        0: Print the statistic\n"
    << "        1: Print contingency table\n"
    << "        2: Print scores in the form: <variant> <score> <c|t>\n"
    << "    -bootstrap <y|n> default: n\n"
    << "        Bootstrap variants\n"
    << "        outfmt will be forced to 0 in bootstrap mode\n"
    << "\n";
}

void parse_arguments(int argc, char const* argv[]) {
  if (argc < 9) {
    usage();
    exit(1);
  }
  int i = 1;
  while (i < argc) {
    if (string(argv[i]).compare("-c") == 0) {
      opts.cfile = argv[++i];
      if (!has_file(opts.cfile)) {
        error_and_exit(__PRETTY_FUNCTION__, "Case SNP file does not exist.");
      }
    } else if (string(argv[i]).compare("-t") == 0) {
      opts.tfile = argv[++i];
      if (!has_file(opts.tfile)) {
        error_and_exit(__PRETTY_FUNCTION__, "Control SNP file does not exist.");
      }
    } else if (string(argv[i]).compare("-v") == 0) {
      opts.vfile = argv[++i];
      if (!has_file(opts.vfile)) {
        error_and_exit(__PRETTY_FUNCTION__, "Variant score file does not exist.");
      }
    } else if (string(argv[i]).compare("-g") == 0) {
      opts.gfile = argv[++i];
      if (!has_file(opts.gfile)) {
        error_and_exit(__PRETTY_FUNCTION__, "Gene score file does not exist.");
      }
    } else if (string(argv[i]).compare("-vcutoff") == 0) {
      opts.vcutoff = stod(argv[++i]);
      if (opts.vcutoff < 0 || opts.vcutoff > 1) {
        error_and_exit(__PRETTY_FUNCTION__, "Probability cutoff must be within in the interval [0, 1].");
      }
    } else if (string(argv[i]).compare("-vmild") == 0) {
      opts.mfile = argv[++i];
      if (!has_file(opts.mfile)) {
        error_and_exit(__PRETTY_FUNCTION__, "Mild mutation file does not exist.");
      }
    } else if (string(argv[i]).compare("-excl") == 0) {
      opts.efile = argv[++i];
      if (!has_file(opts.efile)) {
        error_and_exit(__PRETTY_FUNCTION__, "Gene score file does not exist.");
      }
    } else if (string(argv[i]).compare("-enforce") == 0) {
      opts.sfile = argv[++i];
      if (!has_file(opts.sfile)) {
        error_and_exit(__PRETTY_FUNCTION__, "Gene score file does not exist.");
      }
    } else if (string(argv[i]).compare("-stat") == 0) {
      string statisticStr(argv[++i]);
      if (statisticStr.compare("pvalue") == 0) {
        opts.statistic = PVALUE;
      } else if (statisticStr.compare("auc") == 0) {
        opts.statistic = AUC;
      } else if (statisticStr.compare("dor") == 0) {
        opts.statistic = DOR;
      } else if (statisticStr.compare("lr") == 0) {
        opts.statistic = LR;
      } else {
        usage();
        error_and_exit(__PRETTY_FUNCTION__, "Unknown choice of statistic.");
      }
    } else if (string(argv[i]).compare("-q") == 0) {
      double q = stod(argv[++i]);
      if (q <= 0 || q >= 1) {
        usage();
        error_and_exit(__PRETTY_FUNCTION__, "-q must be in the interval (0, 1).");
      }
      opts.q = q;
    } else if (string(argv[i]).compare("-log") == 0) {
      string yesOrNo = argv[++i];
      if (yesOrNo.compare("Y") == 0 || yesOrNo.compare("y") == 0) {
        opts.log = true;
      } else if (yesOrNo.compare("N") == 0 || yesOrNo.compare("n") == 0) {
        opts.log = false;
      } else {
        string msg = string("Unknown choice of [-log]: [") + yesOrNo + string("].");
        ERRMSG(msg);
      }
    } else if (string(argv[i]).compare("-outfmt") == 0) {
      int choice = stoi(argv[++i]);
      if (choice == 0) {
        opts.outfmt = 0;
      } else if (choice == 1) {
        opts.outfmt = 1;
      } else if (choice == 2) {
        opts.outfmt = 2;
      } else {
        usage();
        error_and_exit(__PRETTY_FUNCTION__, "-outfmt must be 0, 1 or 2.");
      }
    } else if (string(argv[i]).compare("-bootstrap") == 0) {
      string yesOrNo = argv[++i];
      if (yesOrNo.compare("Y") == 0 || yesOrNo.compare("y") == 0) {
        opts.bootstrap = true;
      } else if (yesOrNo.compare("N") == 0 || yesOrNo.compare("n") == 0) {
        opts.bootstrap = false;
      } else {
        string msg = string("Unknown choice of [-bootstrap]: [") + yesOrNo + string("].");
        ERRMSG(msg);
      }
    } else {
      usage();
      error_and_exit(__PRETTY_FUNCTION__, "Unknown option.");
    }
    i += 1;
  }
  if (opts.cfile.empty()) {
    usage();
    error_and_exit(__PRETTY_FUNCTION__, "No case SNP file specified.");
  }
  if (opts.tfile.empty()) {
    usage();
    error_and_exit(__PRETTY_FUNCTION__, "No control SNP file specified.");
  }
  if (opts.vfile.empty()) {
    usage();
    error_and_exit(__PRETTY_FUNCTION__, "No variant score file specified.");
  }
  if (opts.gfile.empty()) {
    usage();
    error_and_exit(__PRETTY_FUNCTION__, "No gene score file specified.");
  }

  if (opts.bootstrap) {
    opts.outfmt = 0;
  }
}

unordered_map<string, double> load_scores(const string& file) {
  ifstream ifs(file, ifstream::in);
  string buf;
  string mut;
  double score;
  unordered_map<string, double> predictions;
  while (ifs.good()) {
    getline(ifs, buf);
    istringstream iss(buf);
    iss >> mut >> score;
    predictions[mut] = score;
  }
  ifs.close();
  return predictions;
}

unordered_map<string, double> combine_scores(
    const vector<string>& muts,
    const unordered_map<string, double>& vScores,
    const unordered_map<string, double>& gScores,
    const unordered_set<string>& exclGenes,
    bool is_log) {
  unordered_map<string, double> score_map;
  for (auto const& mut : muts) {
    string entrez = mut.substr(0, mut.find_first_of("_"));
    if (CONTAINS(exclGenes, entrez)) continue; // skip exclude genes
    if (is_log) {
      if (CONTAINS(vScores, mut) && CONTAINS(gScores, entrez)) {
        score_map[mut] = vScores.at(mut) + gScores.at(entrez);
      } else {
        score_map[mut] = -999999; // -Inf
      }
    } else {
      if (CONTAINS(vScores, mut) && CONTAINS(gScores, entrez)) {
        score_map[mut] = vScores.at(mut) * gScores.at(entrez);
      } else {
        score_map[mut] = 0.0;
      }
    }
  }
  return score_map;
}

int main(int argc, const char* argv[]) {
  srand(time(NULL));
  parse_arguments(argc, argv);
  // load data files
  vector<string> cid_raw = load_items<string>(opts.cfile);
  vector<string> tid_raw = load_items<string>(opts.tfile);

  unordered_map<string, int> id_map;
  vector<string> id_pool;
  for (const auto& id : cid_raw) {
    id_map[id] = 1;
    id_pool.push_back(id);
  }
  for (const auto& id : tid_raw) {
    id_map[id] = 0;
    id_pool.push_back(id);
  }
  int num_repeats = opts.bootstrap ? 100 : 1;
  for (int bindex = 0; bindex < num_repeats; ++bindex) {
    vector<string> cid;
    vector<string> tid;
    if (opts.bootstrap) {
      // bootstrap cid and tid
      for (size_t k = 0; k < id_pool.size(); ++k) {
        int index = rand() % id_pool.size();
        string id = id_pool[index];
        if (id_map.at(id) == 1) {
          cid.push_back(id);
        } else {
          tid.push_back(id);
        }
      }
    } else {
      cid = cid_raw;
      tid = tid_raw;
    }

    unordered_map<string, double> vScores = load_scores(opts.vfile);
    // filter mild mutations/variants
    if (!opts.mfile.empty()) {
      vector<string> mildMutations = load_items<string>(opts.mfile);
      for (auto const& mutation : mildMutations) {
        if (CONTAINS(vScores, mutation)) {
          vScores[mutation] = 0.0;
        }
      }
    }
    // filter scores for variants
    if (opts.vcutoff > 0) {
      for (auto it = vScores.begin(); it != vScores.end(); ++it) {
        if (it->second < opts.vcutoff) {
          it->second = 0.0;
        }
      }
    }
    unordered_map<string, double> gScores = load_scores(opts.gfile);
    // enforce seed gene scores if specified
    if (!opts.sfile.empty()) {
      vector<string> seedGenes = load_items<string>(opts.sfile);
      for (auto const& gene : seedGenes) {
        if (CONTAINS(gScores, gene)) {
          gScores[gene] = opts.log ? 0.0 : 1.0;
        }
      }
    }
    // exclude genes if specified
    unordered_set<string> exclGenes;
    if (!opts.efile.empty()) {
      vector<string> exGenes = load_items<string>(opts.efile);
      for (auto const& gene : exGenes) {
        exclGenes.insert(gene);
      }
    }
    // MutPred score * gene score
    auto cScores = combine_scores(cid, vScores, gScores, exclGenes, opts.log);
    auto tScores = combine_scores(tid, vScores, gScores, exclGenes, opts.log);

    vector<double> collectedTScores;
    for (const auto& score : tScores) collectedTScores.push_back(score.second);
    // determine tail
    double cutoff;
    if (opts.log)
      cutoff = quantile<double>(collectedTScores, opts.q);
    else
      cutoff = max(1e-16, quantile<double>(collectedTScores, opts.q));
    vector<double> pred;
    vector<bool> label;
    uint a(0), b(0), c(0), d(0); // For p-value
    uint TP(0), TN(0), FP(0), FN(0); // For DOR and LR
    // uint num_rt(0), num_case_rt(0); // For LR
    // double p(0.0), q(0.0); // For LR
    switch (opts.statistic) {
      case PVALUE:
        for (auto const& score : tScores) {
          if (score.second < cutoff) {
            a += 1;
          }
        }
        for (auto const& score : cScores) {
          if (score.second < cutoff) {
            b += 1;
          }
        }
        c = tScores.size() - a;
        d = cScores.size() - b;
        if (opts.outfmt == 0) {
          cout << htest::fishertest<>(a, c, b, d, htest::RIGHT) << endl;
        } else if (opts.outfmt == 1) {
          cout << htest::fishertest<>(a, c, b, d, htest::RIGHT) << endl;
          // cout << "cn: " << c << "/" << a+c << " cs: " << d << "/" << b+d << endl;
          cout << "        |     <q |    >=q with q = " << cutoff << "\n";
          cout << "-------------------------\n";
          cout << "control | " << setw(6) << fixed << a << " | " << setw(6) << fixed << c << "\n";
          cout << "case    | " << setw(6) << fixed << b << " | " << setw(6) << fixed << d << "\n";
        } else if (opts.outfmt == 2) {
          for (const auto& score : cScores) cout << score.first << "\t" << score.second << "\tc" << endl;
          for (const auto& score : tScores) cout << score.first << "\t" << score.second << "\tt" << endl;
        }
        break;
      case AUC:
        for (auto const& score : tScores) {
          if (score.second >= cutoff) {
            pred.push_back(score.second);
            label.push_back(false);
          }
        }
        for (auto const& score : cScores) {
          if (score.second >= cutoff) {
            pred.push_back(score.second);
            label.push_back(true);
          }
        }
        cout << get_auc(pred, label) << endl;
        break;
      case LR:
        // // Likelihood Ratio = (q / (1-q)) / (p / (1-p))
        // // where q: the #case / #var in the right tail
        // // and p: the #case / #var overall
        // for (auto const& score : tScores) {
        //   if (score.second >= cutoff) {
        //     num_rt ++;
        //   }
        // }
        // for (auto const& score : cScores) {
        //   if (score.second >= cutoff) {
        //     num_rt ++;
        //     num_case_rt ++;
        //   }
        // }
        // p = (double)(num_case_rt) / (double)(num_rt);
        // q = (double)(cScores.size()) / (double)(cScores.size() + tScores.size());
        // cout << (q / (1.0 - q)) / (p / (1.0 - p)) << endl;
        for (auto const& score : tScores) {
          if (score.second >= cutoff) FP ++;
          else TN ++;
        }
        for (auto const& score : cScores) {
          if (score.second >= cutoff) TP ++;
          else FN ++;
        }
        cout << (double(TP)/double(TP+FN)) / (double(FP)/double(FP+TN)) << endl;
        break;
      case DOR:
        // DOR = (TP * TN) / (FP * FN)
        for (auto const& score : tScores) {
          if (score.second >= cutoff) FP ++;
          else TN ++;
        }
        for (auto const& score : cScores) {
          if (score.second >= cutoff) TP ++;
          else FN ++;
        }
        cout << (double)(TP * TN) / (double)(FP * FN) << endl;
        break;
      default:
        // nop
        break;
    }
  }
  return 0;
}

// -------------
// Yuxiang Jiang (yuxjiang@indiana.edu)
// Department of Computer Science
// Indiana University, Bloomington
// Last modified: Tue 06 Jul 2021 02:33:22 PM E
