"""Generated numerical tables and claim macros with source-evidence entries."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from proceedings_data import ARMS, dispersion, patient_average, strict_mean


def tex(text):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "_": r"\_",
                    "#": r"\#", "{": r"\{", "}": r"\}"}
    return "".join(replacements.get(char, char) for char in str(text))


def number(value, digits=4, signed=False):
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return format(float(value), ("+" if signed else "") + f".{digits}f")


def stability_patient(frame):
    result = []
    for (patient, arm, method), g in frame.groupby(["patient_index", "arm", "method"]):
        valid = len(g) == 3 and bool(g.defined.all())
        row = {"patient_index": patient, "arm": arm, "method": method}
        for metric in ("spearman", "top_five_jaccard", "sign_agreement"):
            row[metric] = float(g[metric].mean()) if valid and g[metric].notna().all() else np.nan
        result.append(row)
    return pd.DataFrame(result)


def balance_comparison(design):
    """Compare distance to 50/50 after averaging all three seeds within patient."""
    means = patient_average(design, ['positive_fraction'])
    paired = means.pivot(index='patient_index', columns='arm', values='positive_fraction')
    if paired.isna().any().any():
        raise ValueError('Incomplete patient pairs for class-balance comparison')
    distance = (paired - .5).abs()
    return {'adaptive_closer_patients': int((distance.adaptive < distance.random).sum()),
            'random_closer_patients': int((distance.random < distance.adaptive).sum()),
            'equal_distance_patients': int((distance.random == distance.adaptive).sum())}


def summarize(data):
    f = patient_average(data["fidelity"], ["novel_mae", "constant_baseline_novel_mae", "all_draws_mae"])
    d = patient_average(data["design"], ["positive_fraction", "duplicate_rows", "unique_masks", "centered_design_condition_number"])
    deletion = patient_average(data["deletion"], ["area_under_curve", "normalized_area"],
                               ("patient_index", "arm", "control"))
    p = data["patients"]
    result = {"patients": data["n"], "runs": data["n"]*3, "images": sum(data["image_counts"]),
        "image_min": min(data["image_counts"]), "image_max": max(data["image_counts"]),
        "model_queries": data["queries"], "primary_patients_available": int(p.primary_available.sum()),
        "primary_patients_unavailable": int((~p.primary_available).sum()),
        "paired_mae": data["aggregate"]["mean_patient_paired_mae_difference"],
        "paired_mae_dispersion": dispersion(p.paired_adaptive_minus_random_mae),
        "adaptive_better": int((p.paired_adaptive_minus_random_mae < 0).sum()),
        "random_better": int((p.paired_adaptive_minus_random_mae > 0).sum()),
        "equal_mae": int((p.paired_adaptive_minus_random_mae == 0).sum()),
        "target_successes": int(data["stages"].balance_reached.sum()),
        "pool_deficit_rows": int(data["stages"].deficit_rows.sum()),
        "biased_rows": int(data["stages"].biased_rows.sum()),
        "one_pool_runs": int(data["stages"].one_pool_fallback.sum()),
        "novel_min": int(data["fidelity"].novel_rows.min()), "novel_max": int(data["fidelity"].novel_rows.max()),
        "one_class_evaluations": int(((data["fidelity"].query('arm == "random"').novel_negative_rows == 0)
                                     | (data["fidelity"].query('arm == "random"').novel_positive_rows == 0)).sum()),
        "balance_comparison": balance_comparison(data['design']),
        "query_counts": {key: int(value) for key, value in
                         data['query_counts'].drop(columns=['patient_index', 'seed']).sum().items()},
        "arms": {}, "deletion": {}}
    result['adaptive_extra_queries'] = (result['query_counts']['adaptive_singletons'] +
                                       result['query_counts']['adaptive_validation'])
    for arm in ARMS:
        rows, design = f[f.arm == arm], d[d.arm == arm]
        raw = data["fidelity"][data["fidelity"].arm == arm]
        raw_design = data["design"][data["design"].arm == arm]
        result["arms"][arm] = {"mae": strict_mean(rows.novel_mae),
            "mae_dispersion": dispersion(rows.novel_mae),
            "constant_mae": strict_mean(rows.constant_baseline_novel_mae),
            "better_than_constant_patients": int((rows.novel_mae < rows.constant_baseline_novel_mae).sum()),
            "better_than_constant_runs": int((raw.novel_mae < raw.constant_baseline_novel_mae).sum()),
            "all_draws_mae": strict_mean(rows.all_draws_mae),
            "positive_fraction": strict_mean(design.positive_fraction),
            "both_prediction_classes_runs": int(((raw_design.positive_fraction > 0) &
                                                  (raw_design.positive_fraction < 1)).sum()),
            "duplicate_fraction": strict_mean(design.duplicate_rows)/1000,
            "unique_masks_mean": strict_mean(design.unique_masks),
            "singular_designs": int(raw_design.centered_design_singular.sum()),
            "condition_median": dispersion(design.centered_design_condition_number)["median"],
            "out_of_range_novel_scores": int(raw.novel_out_of_range_scores.sum())}
        a = deletion[deletion.arm == arm].pivot(index="patient_index", columns="control", values="area_under_curve")
        norm = deletion[deletion.arm == arm].pivot(index="patient_index", columns="control", values="normalized_area")
        result["deletion"][arm] = {control: strict_mean(a[control]) for control in ("descending","ascending","random")}
        result["deletion"][arm].update(descending_minus_random=strict_mean(a.descending-a.random),
            descending_minus_ascending=strict_mean(a.descending-a.ascending),
            descending_below_random_patients=int((a.descending < a.random).sum()),
            delta_dispersion=dispersion(a.descending-a.random),
            normalized_descending_minus_random=strict_mean(norm.descending-norm.random))
    return result


def export_tables(main, pilot, output):
    output = Path(output);output.mkdir(parents=True,exist_ok=True)
    primary = summarize(main); pilot_metrics = summarize(pilot)
    evidence = {entry["path"]: entry for entry in main["evidence"] + pilot["evidence"]}
    ledger = {"schema": 1, "phase": main["population"], "analysis_unit": "patient mean across all three seeds",
        "dispersion": "Sample SD (ddof=1); median and linearly interpolated quartiles across patient means; no inferential CI",
        "sources": list(evidence.values()), "entries": []}
    def entry(identifier, definition, values, population=None):
        ledger["entries"].append({"id": identifier, "definition": definition,
            "population": population or main["population"], "values": values,
            "source_set": "pilot" if population == "pilot" else "primary", "validation": "passed"})
    def table(name, caption, headers, rows, definition, population=None):
        # First column wraps; numerical columns are aligned and source-generated.
        widths = "@{}>{\\raggedright\\arraybackslash}X" + "r"*(len(headers)-1) + "@{}"
        lines = [r"\begin{table}[!htbp]",r"\centering\small",r"\caption{"+caption+r"}",
            r"\label{tab:"+name+r"}",r"\begin{tabularx}{\linewidth}{"+widths+r"}",r"\toprule",
            " & ".join(tex(h) for h in headers)+r" \\",r"\midrule"]
        lines += [" & ".join(tex(v) for v in row)+r" \\" for row in rows]
        lines += [r"\bottomrule",r"\end{tabularx}",r"\end{table}"]
        (output/(name+".tex")).write_text("\n".join(lines)+"\n")
        pd.DataFrame(rows,columns=headers).to_csv(output/(name+".csv"),index=False)
        entry("table:"+name,definition,{"headers":headers,"rows":rows},population)
    n, runs = primary["patients"], primary["runs"]
    table("design", "Cohort and frozen experimental design. The ten-patient convenience pilot was inspected before expansion; all analyses are exploratory.",
        ["Item", "Value"], [
        ("Primary analysis population", f"{n} patients; {runs} patient–seed runs"),
        ("Images in primary population",f"{primary['images']}; {primary['image_min']}–{primary['image_max']} per patient"),
        ("Eligible expansion population", "135 positive test09 patients; 3,072 images"),
        ("Eligibility", "Observed liver-fat label > 0; ≥20 images; complete coverage"),
        ("Prediction model", "DenseNet121 + supplied SETNET_GAT, split09"),
        ("Graph edges", "Feature correlation >0.95; graph rebuilt per subset"),
        ("Training per sampling arm", "1,000 rows; subset size uniform on 3,…,n"),
        ("Seeds / evaluation", "0,1,2 / 200 shared draws per patient–seed"),
        ("Response / Ridge", "Class-1 softmax probability / alpha=1, intercept"),
        ("Elastic Net (pilot only)", "Five-fold training-only CV; unscaled binary design"),
        ("Primary inference calls",f"{primary['model_queries']:,} (adaptive overhead included)")],
        "Verified plans, image counts, raw ledgers and fixed scientific settings; eligibility expansion manifest")
    rows=[]
    for data, m in [(main,primary)] + ([(pilot,pilot_metrics)] if main["population"]=="full" else []):
        prefix="Full" if data["population"]=="full" else "Pilot"
        for arm in ARMS:
            a=m["arms"][arm]
            rows.append((prefix+": "+arm.capitalize(),number(a["mae"]),number(a["mae_dispersion"]["sd"]),number(a["constant_mae"])))
        rows.append((prefix+": adaptive − random",number(m["paired_mae"],signed=True),
                     number(m["paired_mae_dispersion"]["sd"]),"—"))
    table("fidelity", "Primary shared-novel probability fidelity. Means and sample SDs summarize patient means over three seeds. Lower MAE is better; each constant is its arm's training-mean probability.",
        ["Population / method", "Mean MAE", "Patient SD", "Constant MAE"],rows,
        "Shared-novel MAE and own training-mean constant; average all three seeds within patient, then patients equally")
    table("fidelity_counts", "Fidelity availability and paired directions. Strictly lower error defines a favorable comparison; unavailable evaluations are not assigned zero.",
        ["Primary-population diagnostic", "Count"], [
        ("Patients favoring random / adaptive / equal",f"{primary['random_better']} / {primary['adaptive_better']} / {primary['equal_mae']}"),
        ("Patients with / without all primary seed pairs",f"{primary['primary_patients_available']} / {primary['primary_patients_unavailable']}"),
        ("Shared-novel rows per patient–seed",f"{primary['novel_min']}–{primary['novel_max']}"),
        ("One-class shared-novel sets",f"{primary['one_class_evaluations']} / {runs}"),
        *[(arm.capitalize()+": patients beating own constant",f"{primary['arms'][arm]['better_than_constant_patients']} / {n}") for arm in ARMS],
        *[(arm.capitalize()+": out-of-range novel scores",str(primary['arms'][arm]['out_of_range_novel_scores'])) for arm in ARMS]],
        "Availability, shared evaluation support, strict paired directions and untruncated score counts")
    s=main["stages"]; pools=main["pool_rows"]
    rows=[("Requested positive-prediction proportion","0.5000","0.5000"),
          ("Achieved proportion (mean patient)",*[number(primary["arms"][a]["positive_fraction"]) for a in ARMS]),
          ("Training sets with both predicted classes",*[f"{primary['arms'][a]['both_prediction_classes_runs']} / {runs}" for a in ARMS]),
          ("Unique masks / 1,000 (mean patient)",*[number(primary["arms"][a]["unique_masks_mean"],1) for a in ARMS]),
          ("Duplicate rows (%)",*[number(primary["arms"][a]["duplicate_fraction"]*100,2) for a in ARMS]),
          ("Rank-deficient centered designs",*[f"{primary['arms'][a]['singular_designs']} / {runs}" for a in ARMS]),
          ("Median patient-mean condition number",*[number(primary["arms"][a]["condition_median"] ,2) for a in ARMS]),
          ("Adaptive target successes / failures","N/A",f"{primary['target_successes']} / {runs-primary['target_successes']}"),
          ("Biased draws needing reallocation","N/A",f"{primary['pool_deficit_rows']:,} / {primary['biased_rows']:,}"),
          ("One-pool fallback runs","N/A",str(primary['one_pool_runs']))]
    # Random has no prescribed class target: do not imply it requested 50/50.
    rows[0]=("Requested positive-prediction proportion","Not targeted","0.5000")
    for prop in (.85,.15):
        g=pools[np.isclose(pools.requested_positive_proportion,prop)] if len(pools) else pools
        realized=g.groupby(["patient_index","seed"]).realized_positive_fraction.mean() if len(g) else pd.Series(dtype=float)
        patient_realized=realized.groupby(level='patient_index').mean() if len(realized) else pd.Series(dtype=float)
        count=len(patient_realized)
        rows.append((f"Intended {int(prop*100)}/{int((1-prop)*100+.1)} pool ratio: realized positive fraction","N/A",
                     number(patient_realized.mean())+f" ({count} {'patient' if count==1 else 'patients'})"))
    table("sampling", "Sampling feasibility and realized behavior. Pool ratios concern singleton-predicted image classes, not observed disease labels or guaranteed subset predictions. Pool-composition means first average available runs within patient, then patients within each requested-ratio stratum.",
          ["Diagnostic", "Random", "Adaptive"],rows,
          "Validated binary-design SVD and exact reconstruction of adaptive stage/pool clamping; duplicates = rows minus unique masks")
    rows=[]
    for arm in ARMS:
        a=primary["deletion"][arm]
        rows.extend([(arm.capitalize()+": descending",number(a["descending"])),
                     (arm.capitalize()+": ascending",number(a["ascending"])),
                     (arm.capitalize()+": shared random",number(a["random"])),
                     (arm.capitalize()+": descending − random",number(a["descending_minus_random"],signed=True)),
                     (arm.capitalize()+": descending − ascending",number(a["descending_minus_ascending"],signed=True)),
                     (arm.capitalize()+": patients below random",f"{a['descending_below_random_patients']} / {n}")])
    table("deletion", "Deletion-based model behavior: mean raw probability AUC over actual deleted fractions. Negative descending-minus-control differences indicate a lower class-1 trajectory; they do not establish clinical importance.",
          ["Ranking / paired contrast", "Patient mean AUC / count"],rows,
          "Trapezoidal integration on actual patient-specific deletion fractions; seeds averaged before equal patient weighting")
    enet=patient_average(pilot["enet"],["novel_mae"],("patient_index","arm","method"))
    stab=stability_patient(pilot["stability"])
    rows=[]
    secondary={}
    for arm in ARMS:
        e=enet[enet.arm==arm].pivot(index="patient_index",columns="method",values="novel_mae")
        delta=strict_mean(e.elastic_net-e.ridge)
        secondary[arm]={"enet_minus_ridge_mae":delta}
        rows.append((arm.capitalize()+": Elastic Net − Ridge MAE",number(delta,signed=True),"10 / 10"))
        for method,label in (("ridge","Ridge"),("elastic_net","Elastic Net"),("marginal_correlation","Pearson")):
            g=stab[(stab.arm==arm)&(stab.method==method)]
            m=dispersion(g.spearman)
            secondary[arm][method]={"stability":m}
            rows.append((arm.capitalize()+": "+label+" seed Spearman",number(m['mean']),f"{m['n']} / 10"))
            j=dispersion(g.top_five_jaccard)
            rows.append((arm.capitalize()+": "+label+" top-5 Jaccard",number(j['mean']),f"{j['n']} / 10"))
            sign=dispersion(g.sign_agreement)
            rows.append((arm.capitalize()+": "+label+" sign agreement",number(sign['mean']),f"{sign['n']} / 10"))
            l=pilot['loo_agreement'];l=l[(l.arm==arm)&(l.method==method)]
            z=dispersion(l.spearman)
            secondary[arm][method]['loo_spearman']=z
            rows.append((arm.capitalize()+": "+label+" versus LOO Spearman",number(z['mean']),f"{z['n']} / 10"))
    loo_means=pilot['loo'].assign(absolute_change=pilot['loo'].delta_class1.abs()).groupby('patient_index').absolute_change.mean()
    rows.append(('LOO: mean absolute probability change',number(loo_means.mean()),f'{len(loo_means)} / 10'))
    secondary['loo_mean_absolute_change']=dispersion(loo_means)
    table("secondary", "Additional ten-patient analyses only. Seed stability averages all three seed-pair comparisons within each patient. LOO agreement averages three coefficient/ranking-versus-LOO correlations within patient; constant vectors are unavailable.",
          ["Pilot diagnostic", "Mean", "Defined / total"],rows,
          "Training-only CV probability regressions; complete-seed ranking stability; descriptive rank agreement with saved class-1 LOO effects", "pilot")
    entry("claims:primary", "All primary descriptive numerical claims", primary)
    main['query_counts'].to_csv(output/'query_counts_by_seed.csv',index=False)
    entry('queries:stages', 'Each preserved query ledger reconciles stage counts, masks and responses; shared evaluation and random-deletion calls counted once', primary['query_counts'])
    # Availability is part B of the fidelity table, keeping five numbered tables.
    counts_text=(output/'fidelity_counts.tex').read_text()
    counts_tabular=counts_text[counts_text.index(r'\begin{tabularx}'):counts_text.index(r'\end{tabularx}')+len(r'\end{tabularx}')]
    fidelity_text=(output/'fidelity.tex').read_text()
    fidelity_text=fidelity_text.replace(r'\end{table}',
        '\\par\\vspace{0.8em}\\textit{Availability and paired directions: primary population}\\par\n'
        +r'\label{tab:fidelity_counts}'+'\n'+counts_tabular+'\n'+r'\end{table}')
    (output/'fidelity.tex').write_text(fidelity_text)
    ledger['entries'].append({'id':'table:fidelity:availability','definition':'Part B of the primary fidelity table; see table:fidelity_counts for generated values.',
                              'validation':'passed','source_set':'primary','population':main['population']})
    entry("claims:pilot", "Separate convenience pilot results", pilot_metrics, "pilot")
    entry("claims:secondary", "Ten-patient secondary comparisons and availability", secondary, "pilot")
    macros={"BuildPopulation":f"{'Full eligible-cohort analysis' if main['population']=='full' else 'Ten-patient exploratory pilot; full-cohort results pending'}",
        "PrimaryN":str(n),"PrimaryRuns":str(runs),"PrimaryImages":str(primary['images']),
        "RandomMAE":number(primary['arms']['random']['mae'],6),"AdaptiveMAE":number(primary['arms']['adaptive']['mae'],6),
        "PairedMAE":number(primary['paired_mae'],6,True),"RandomFavored":str(primary['random_better']),
        "AdaptiveFavored":str(primary['adaptive_better']),"TargetSuccesses":str(primary['target_successes']),
        "DeficitRows":f"{primary['pool_deficit_rows']:,}","BiasedRows":f"{primary['biased_rows']:,}",
        "PrimaryQueries":f"{primary['model_queries']:,}","NovelMin":str(primary['novel_min']),"NovelMax":str(primary['novel_max']),
        "OneClassSets":str(primary['one_class_evaluations']),
        "PilotRandomMAE":number(pilot_metrics['arms']['random']['mae'],6),
        "PilotAdaptiveMAE":number(pilot_metrics['arms']['adaptive']['mae'],6),
        "PilotPairedMAE":number(pilot_metrics['paired_mae'],6,True),
        "PilotRandomFavored":str(pilot_metrics['random_better']),
        "PilotRuns":str(pilot_metrics['runs']),"PilotQueries":f"{pilot_metrics['model_queries']:,}",
        "PilotLOOQueries":str(len(pilot['loo'])+pilot['n']),
        "PilotTargetSuccesses":str(pilot_metrics['target_successes']),
        "PilotDeficitRows":f"{pilot_metrics['pool_deficit_rows']:,}","PilotBiasedRows":f"{pilot_metrics['biased_rows']:,}",
        "PilotDesigns":str(len(pilot['design'])),"PilotOneClassSets":str(pilot_metrics['one_class_evaluations']),
        "PilotRandomEnetStabilityN":str(secondary['random']['elastic_net']['stability']['n']),
        "PilotRandomEnetDelta":number(secondary['random']['enet_minus_ridge_mae'],6,True),
        "PilotAdaptiveEnetDelta":number(secondary['adaptive']['enet_minus_ridge_mae'],6,True),
        "RandomDeletionDelta":number(primary['deletion']['random']['descending_minus_random'],6,True),
        "AdaptiveDeletionDelta":number(primary['deletion']['adaptive']['descending_minus_random'],6,True)}
    macros.update({
        'BalanceCloser': str(primary['balance_comparison']['adaptive_closer_patients']),
        'MAETies': str(primary['equal_mae']),
        'PairedMedian': number(primary['paired_mae_dispersion']['median'],6,True),
        'PairedQOne': number(primary['paired_mae_dispersion']['q1'],6),
        'PairedQThree': number(primary['paired_mae_dispersion']['q3'],6),
        'AdaptiveExtraQueries': f"{primary['adaptive_extra_queries']:,}",
        'SingletonQueries': f"{primary['query_counts']['adaptive_singletons']:,}",
        'ConstructorQueries': f"{primary['query_counts']['adaptive_validation']:,}",
        'TrainingQueriesPerArm': f"{primary['query_counts']['random_training']:,}",
        'DeficitPercent': number(100*primary['pool_deficit_rows']/primary['biased_rows'],2),
        'FullMissingPatients': str(primary['primary_patients_unavailable']),
    })
    for arm in ARMS:
        prefix=arm.capitalize(); values=primary['arms'][arm]
        macros.update({prefix+'PositivePercent':number(100*values['positive_fraction'],2),
                       prefix+'BothClasses':str(values['both_prediction_classes_runs']),
                       prefix+'DuplicatePercent':number(100*values['duplicate_fraction'],2),
                       prefix+'ConstantMAE':number(values['constant_mae'],6),
                       prefix+'BeatsConstant':str(values['better_than_constant_patients']),
                       prefix+'DeletionDescending':number(primary['deletion'][arm]['descending'],6),
                       prefix+'DeletionAscending':number(primary['deletion'][arm]['ascending'],6),
                       prefix+'DeletionRandom':number(primary['deletion'][arm]['random'],6),
                       prefix+'DeletionBelowControl':str(primary['deletion'][arm]['descending_below_random_patients'])})
    (output/'results.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+tex(v)+'}' for k,v in macros.items())+'\n')
    entry('macros:results','Every generated manuscript numerical macro',macros)
    (output/'metrics.json').write_text(json.dumps({'primary':primary,'pilot':pilot_metrics,'secondary':secondary},indent=2,allow_nan=False)+'\n')
    comparisons=[]
    for label, get in [
        ('Patients',lambda m:m['patients']),('Patient–seed runs',lambda m:m['runs']),
        ('Random class-1 proportion',lambda m:m['arms']['random']['positive_fraction']),
        ('Adaptive class-1 proportion',lambda m:m['arms']['adaptive']['positive_fraction']),
        ('Random Ridge MAE',lambda m:m['arms']['random']['mae']),
        ('Adaptive Ridge MAE',lambda m:m['arms']['adaptive']['mae']),
        ('Adaptive minus random MAE',lambda m:m['paired_mae']),
        ('Patients favoring random',lambda m:m['random_better']),
        ('Patients favoring adaptive',lambda m:m['adaptive_better']),
        ('Equal-MAE patients',lambda m:m['equal_mae']),
        ('Exact adaptive target successes',lambda m:m['target_successes']),
        ('Random duplicate fraction',lambda m:m['arms']['random']['duplicate_fraction']),
        ('Adaptive duplicate fraction',lambda m:m['arms']['adaptive']['duplicate_fraction']),
        ('Random descending minus random AUC',lambda m:m['deletion']['random']['descending_minus_random']),
        ('Adaptive descending minus random AUC',lambda m:m['deletion']['adaptive']['descending_minus_random']),
        ('Primary model queries',lambda m:m['model_queries'])]:
        comparisons.append({'Metric':label,'Full':get(primary),'Pilot':get(pilot_metrics)})
    pd.DataFrame(comparisons).to_csv(output/'full_vs_pilot.csv',index=False)
    entry('comparison:full_vs_pilot','Separate descriptive summaries; the full cohort includes the pilot and is not independent replication',comparisons)
    # The full build adds figure entries and source/build metadata before final export.
    return ledger, primary, pilot_metrics, secondary
