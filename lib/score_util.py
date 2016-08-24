from __future__ import division
import numpy as np

def fast_hist(a, b, n):
    k = np.where((a >= 0) & (a < n))[0]
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def report_scores(meta, hist):
    print '>>>', meta
    # overall accuracy
    acc = np.diag(hist).sum() / hist.sum()
    print 'overall accuracy', acc
    # per-class accuracy
    cl_acc = np.nanmean(np.diag(hist) / hist.sum(1))
    print 'mean accuracy', cl_acc
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print 'mean IU', np.nanmean(iu)
    freq = hist.sum(1) / hist.sum()
    fw_iu = (freq[freq > 0] * iu[freq > 0]).sum()
    print 'fwavacc', fw_iu
    return acc, cl_acc, np.nanmean(iu[iu>0]), fw_iu

def get_scores(hist):
    acc = np.diag(hist).sum() / hist.sum()

    # per-class accuracy
    cl_acc = np.nanmean(np.diag(hist) / hist.sum(1))

    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    freq = hist.sum(1) / hist.sum()
    fw_iu = (freq[freq > 0] * iu[freq > 0]).sum()

    return acc, cl_acc, np.nanmean(iu), fw_iu

def score_out_gt(out, gt, n_cl=None):
    """
    Score output against groundtruth for # classes.
    """
    assert(n_cl is not None)
    hist = fast_hist(gt.flatten(), out.flatten(), n_cl)
    return hist

def score_out_gt_bdry(out, gt, bdry, n_cl=None):
    """
    Score output against groundtruth on boundary for # classes.
    """
    assert(n_cl is not None)
    hist = fast_hist(gt[0, bdry], out[bdry], n_cl)
    return hist
