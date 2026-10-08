# Follow-up prompt: determine whether the missing images exist

Copy the prompt below into the agent on the original server. You do not need to know the image locations first. If convenient, give that agent the private request ZIP prepared at `outputs/reproducibility/next-run-readiness/original-server-image-request.zip`. The prompt also works without that ZIP: the agent can inventory the original metadata/test split first. Keep the ZIP and patient-level reports private.

---

We have successfully run the transferred ultrasound GNN and both random/adaptive sampling on Dartmouth Discovery. We have one complete eligible patient. We need to determine whether additional original cropped images are still available; their existence and location are currently unknown. Please investigate the original server and prepare a private availability report, plus a transfer bundle only for files actually found.

This is an image-availability investigation. Do not repeat environment/provenance captures already supplied, run models or fitting, retrain, cluster, modify source/environments, change metadata/splits, delete files or overwrite results. You may create new private reports and copy verified files into a new bundle. Do not send data to any remote service or publish it to GitHub.

1. Establish the search targets without requiring me to locate directories manually.
   - If the private request ZIP is available, inspect/extract it into a new private working directory. It contains `pilot_upload_images.csv` (152 missing filenames for nine additional patients), `pilot_upload_patients.csv` (the ten planned pilot patients), `eligible_test09_patients.csv` (the 135-patient target cohort and complete image lists), and `all_missing_test09_images.csv` (the broader destination's 14,940 missing test-image references). Its manifest records hashes and destination metadata/split identity. Prioritize pilot completion; do not assume those selected files exist.
   - If the ZIP is absent, start with `/home/liuusa_tw/twbabd_image_xai_20062024/` and the dataset09 metadata/test CSVs already used by this project, adjusting the root if moved. Inventory all test09 patients with `liver_fatty > 0` and at least 20 metadata images. The destination has one complete patient; your initial inventory can proceed without knowing its ID or the destination's exact missing list. Report which metadata/split files and hashes you used so we can check that cohorts match.

2. Search likely locations in stages, recording what was actually searched.
   - Start with `/home/liuusa_tw/data/cropped_images/`, the original project, its configured image path, earlier private transfer bundles, and relevant dataset/backup locations referenced by source or existing transfer scripts/logs. Read the image-path setting internally if necessary; never export `.env` contents or credentials.
   - Use filename inventories such as `rg --files --hidden --no-ignore <explicit-roots>` before opening images. Look for exact `<MI_ID>_<IMG_ID>.jpg` names, nested patient/image layouts, moved directories, and symlinked dataset locations. Record unresolved symlinks, unavailable mounts and permission failures separately from a confirmed absent filename in a searched directory.
   - Inspect listings of relevant ZIP/tar archives for matching names without blindly extracting entire datasets. Search additional accessible dataset/backup roots suggested by evidence; do not scan unrelated users' storage or launch an unbounded whole-server crawl. If needed, report which offline backup/mount would require separate access.
   - Search copies of original images in organized result directories as candidates. Ranked thumbnails, plots, mosaics, screenshots, notebook displays and embeddings are not replacements for the original cropped image. An alternate extension/name needs evidence of the original patient/image association and exact crop; record it as an unresolved candidate until verified. Do not regenerate crops from raw ultrasounds in this investigation.

3. Verify candidate originals before treating them as recovered.
   - Keep the patient/image association and metadata image lists unchanged. Check readability with Pillow, image format/dimensions, file size and SHA-256. Record absolute source paths privately. Do not load model pickle/checkpoint files.
   - If multiple files claim the same patient/image ID, compare hashes. Identical copies can be deduplicated in a transfer bundle while preserving all source-path provenance. Different hashes require a conflict report and source/preprocessing evidence; do not choose one silently.
   - Report full-patient coverage against the entire metadata image list, including files already on the destination when its manifest is available. Locating 152 candidate files is not proof of ten complete, correctly matched patients.

4. Prepare actionable private outputs.
   - `image_availability.csv`: requested IDs/filenames, status (`found_verified`, `not_found_in_searched_locations`, `unresolved_candidate`, `conflicting_copies`, or `inaccessible_location`), source path, SHA-256, bytes, image format/dimensions and notes.
   - `patient_coverage.csv`: eligible patient, required image count, verified available count, unresolved/missing count and whether the patient is complete. Keep patient identifiers out of the chat summary.
   - `search_report.md`/JSON: roots/archives searched, exclusions, inaccessible locations, metadata/split hashes and aggregate findings. Say "not found in the searched locations" when that is what the evidence establishes; do not claim a file is permanently lost.
   - If all pilot targets cannot be recovered, identify alternative complete eligible patients from the same cohort. Order alternatives by image count and patient ID, independently of predictions/explanation strength. Report their entire image lists so the destination can reconcile existing versus additional files. Any change to the planned pilot cohort must be explicit.
   - Copy only verified required original JPEGs into a new private bundle, using the destination's canonical filenames only when their identity is established. Preserve originals. Include a mapping from original source paths to bundle paths with hashes and byte sizes; hash payloads without self-referential manifest entries. Do not retransmit existing models/results/environments or automatically transfer the bundle over the network.

5. Return a concise aggregate summary: how many requested images were found/verified, whether nine additional complete patients are recoverable, whether alternative complete patients exist, what remains inaccessible/unknown, and the private report/bundle locations. If no additional complete patient is recoverable, that is a useful result; do not fabricate replacements or alter the cohort to manufacture completeness.

---

## Work that can proceed while this search runs

The ten-patient pilot is a staging target. A clearly labeled **one-patient methods study** can proceed with the complete patient's original images: sampler debugging, seeded random/adaptive comparisons, independent-mask surrogate fidelity and node-deletion diagnostics. More seeds or perturbations measure within-patient variability; the number of independent patients remains one. The current smoke CLI is capped at 20 perturbations; a larger methods study needs its own validated runner and output namespace.

Existing historical prediction/design tables also permit table-only surrogate reanalysis, after separating fitting from image-dependent plotting and addressing fitter/runtime issues. Those stored tables do not provide new GNN responses for unseen masks and cannot replace images for new sampling or deletion experiments. Keep July and November runs and prospective versus historical estimators separate.

Patient-population claims, comparative effectiveness across patients and the ten-/135-patient fresh-inference experiments still require complete input coverage for their stated cohort. Do not silently drop missing image nodes and present the resulting graph as that patient's original graph. If the original images cannot be recovered, report the narrower available cohort and claims explicitly.
