$ErrorActionPreference = "Stop"

$train = Import-Csv 'data\splits\experiment2_train.csv'
$val   = Import-Csv 'data\splits\experiment2_val.csv'
$test  = Import-Csv 'data\splits\experiment2_test.csv'

Write-Output "=== 1. FLAT extracted_dataset PATH VALIDATION ==="
$train_img_ok = 0; $train_mask_ok = 0
foreach ($row in $train) {
    if (Test-Path ("extracted_dataset\images\" + $row.filename)) { $train_img_ok++ }
    if (Test-Path ("extracted_dataset\masks\" + $row.filename)) { $train_mask_ok++ }
}
Write-Output "Train: $train_img_ok / $($train.Count) images, $train_mask_ok / $($train.Count) masks"

$val_img_ok = 0; $val_mask_ok = 0
foreach ($row in $val) {
    if (Test-Path ("extracted_dataset\images\" + $row.filename)) { $val_img_ok++ }
    if (Test-Path ("extracted_dataset\masks\" + $row.filename)) { $val_mask_ok++ }
}
Write-Output "Val:   $val_img_ok / $($val.Count) images, $val_mask_ok / $($val.Count) masks"

$test_img_ok = 0; $test_mask_ok = 0
foreach ($row in $test) {
    if (Test-Path ("extracted_dataset\images\" + $row.filename)) { $test_img_ok++ }
    if (Test-Path ("extracted_dataset\masks\" + $row.filename)) { $test_mask_ok++ }
}
Write-Output "Test:  $test_img_ok / $($test.Count) images, $test_mask_ok / $($test.Count) masks"

Write-Output ""
Write-Output "=== 2. PARTITIONED Part_01..Part_10 PATH VALIDATION ==="
$part_images = @{}
$part_masks = @{}
for ($p = 1; $p -le 10; $p++) {
    $pname = "Part_{0:D2}" -f $p
    $p_img_dir = "Oceantrace_Colab_Exp2_UploadParts\$pname\extracted_dataset\images"
    $p_msk_dir = "Oceantrace_Colab_Exp2_UploadParts\$pname\extracted_dataset\masks"
    if (Test-Path $p_img_dir) {
        Get-ChildItem -Path $p_img_dir -Filter *.tif | ForEach-Object { $part_images[$_.Name] = $_.FullName }
    }
    if (Test-Path $p_msk_dir) {
        Get-ChildItem -Path $p_msk_dir -Filter *.tif | ForEach-Object { $part_masks[$_.Name] = $_.FullName }
    }
}
Write-Output "Indexed $($part_images.Count) images and $($part_masks.Count) masks across Part_01..Part_10"

$part_train_img_ok = 0; $part_train_mask_ok = 0
foreach ($row in $train) {
    if ($part_images.ContainsKey($row.filename)) { $part_train_img_ok++ }
    if ($part_masks.ContainsKey($row.filename)) { $part_train_mask_ok++ }
}
Write-Output "Partitioned Train: $part_train_img_ok / $($train.Count) images, $part_train_mask_ok / $($train.Count) masks"

$part_val_img_ok = 0; $part_val_mask_ok = 0
foreach ($row in $val) {
    if ($part_images.ContainsKey($row.filename)) { $part_val_img_ok++ }
    if ($part_masks.ContainsKey($row.filename)) { $part_val_mask_ok++ }
}
Write-Output "Partitioned Val:   $part_val_img_ok / $($val.Count) images, $part_val_mask_ok / $($val.Count) masks"

$part_test_img_ok = 0; $part_test_mask_ok = 0
foreach ($row in $test) {
    if ($part_images.ContainsKey($row.filename)) { $part_test_img_ok++ }
    if ($part_masks.ContainsKey($row.filename)) { $part_test_mask_ok++ }
}
Write-Output "Partitioned Test:  $part_test_img_ok / $($test.Count) images, $part_test_mask_ok / $($test.Count) masks"

Write-Output ""
Write-Output "=== 3. 00720 (1).tif CHECK ==="
$dup_in_train = ($train | Where-Object { $_.filename -like '*00720*1*' -or $_.filename -eq '00720 (1).tif' }).Count
$dup_in_val   = ($val | Where-Object { $_.filename -like '*00720*1*' -or $_.filename -eq '00720 (1).tif' }).Count
$dup_in_test  = ($test | Where-Object { $_.filename -like '*00720*1*' -or $_.filename -eq '00720 (1).tif' }).Count
Write-Output "00720 (1).tif in Train: $dup_in_train"
Write-Output "00720 (1).tif in Val:   $dup_in_val"
Write-Output "00720 (1).tif in Test:  $dup_in_test"

Write-Output ""
Write-Output "=== 4. LEAKAGE AUDIT (FILENAMES AND PARENT ACQUISITIONS) ==="
$train_files = New-Object 'System.Collections.Generic.HashSet[string]'
$train_parents = New-Object 'System.Collections.Generic.HashSet[string]'
foreach ($r in $train) { [void]$train_files.Add($r.filename); [void]$train_parents.Add($r.parent_acquisition_id) }

$val_files = New-Object 'System.Collections.Generic.HashSet[string]'
$val_parents = New-Object 'System.Collections.Generic.HashSet[string]'
foreach ($r in $val) { [void]$val_files.Add($r.filename); [void]$val_parents.Add($r.parent_acquisition_id) }

$test_files = New-Object 'System.Collections.Generic.HashSet[string]'
$test_parents = New-Object 'System.Collections.Generic.HashSet[string]'
foreach ($r in $test) { [void]$test_files.Add($r.filename); [void]$test_parents.Add($r.parent_acquisition_id) }

$file_overlap_tv = 0; foreach ($f in $train_files) { if ($val_files.Contains($f)) { $file_overlap_tv++ } }
$file_overlap_tt = 0; foreach ($f in $train_files) { if ($test_files.Contains($f)) { $file_overlap_tt++ } }
$file_overlap_vt = 0; foreach ($f in $val_files) { if ($test_files.Contains($f)) { $file_overlap_vt++ } }

$parent_overlap_tv = 0; foreach ($p in $train_parents) { if ($val_parents.Contains($p)) { $parent_overlap_tv++ } }
$parent_overlap_tt = 0; foreach ($p in $train_parents) { if ($test_parents.Contains($p)) { $parent_overlap_tt++ } }
$parent_overlap_vt = 0; foreach ($p in $val_parents) { if ($test_parents.Contains($p)) { $parent_overlap_vt++ } }

Write-Output "Train files: $($train_files.Count), Train parents: $($train_parents.Count)"
Write-Output "Val files:   $($val_files.Count), Val parents:   $($val_parents.Count)"
Write-Output "Test files:  $($test_files.Count), Test parents:  $($test_parents.Count)"
Write-Output "Filename overlaps: Train-Val=$file_overlap_tv, Train-Test=$file_overlap_tt, Val-Test=$file_overlap_vt"
Write-Output "Parent overlaps:   Train-Val=$parent_overlap_tv, Train-Test=$parent_overlap_tt, Val-Test=$parent_overlap_vt"
