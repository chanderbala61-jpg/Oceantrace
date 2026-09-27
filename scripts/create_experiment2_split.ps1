param(
    [string]$MetadataIndex = "metadata_index.csv",
    [string]$OutputDir = "data\splits",
    [int]$Seed = 42
)

# 1. Load Metadata
Write-Output "Loading metadata from $MetadataIndex..."
$records = Import-Csv -Path $MetadataIndex

$totalImages = $records.Count
$groups = $records | Group-Object -Property parent_acquisition_id

$numAcquisitions = $groups.Count
Write-Output "Loaded $totalImages images across $numAcquisitions parent acquisitions."

# 2. Group Distribution Analysis
$counts = @($groups | ForEach-Object { $_.Count } | Sort-Object)
$minGroup = $counts[0]
$maxGroup = $counts[-1]

$sum = 0
foreach ($c in $counts) { $sum += $c }
$meanGroup = [Math]::Round(($sum / $counts.Count), 2)

$mid = [Math]::Floor($counts.Count / 2)
if ($counts.Count % 2 -eq 0) {
    $medianGroup = ($counts[$mid - 1] + $counts[$mid]) / 2
} else {
    $medianGroup = $counts[$mid]
}

Write-Output "--- Group Distribution Analysis ---"
Write-Output "Total Images: $totalImages"
Write-Output "Total Parent Acquisitions: $numAcquisitions"
Write-Output "Min Group Size: $minGroup"
Write-Output "Max Group Size: $maxGroup"
Write-Output "Mean Group Size: $meanGroup"
Write-Output "Median Group Size: $medianGroup"

$largeGroups = @($groups | Where-Object { $_.Count -ge 10 } | Sort-Object Count -Descending)
Write-Output "Unusually large groups (count >= 10): $($largeGroups.Count)"
foreach ($lg in $largeGroups) {
    Write-Output "  $($lg.Name): $($lg.Count) images"
}

# 3. Deterministic Group-Level Splitting (Seed = $Seed)
$rand = New-Object System.Random($Seed)

# Convert groups to array and Fisher-Yates shuffle
$groupArray = [System.Collections.ArrayList]@($groups)
for ($i = $groupArray.Count - 1; $i -gt 0; $i--) {
    $j = $rand.Next($i + 1)
    $temp = $groupArray[$i]
    $groupArray[$i] = $groupArray[$j]
    $groupArray[$j] = $temp
}

# Target image counts: Train ~70% (840), Val ~15% (180), Test ~15% (180)
$targetTrainImages = [Math]::Round($totalImages * 0.70)
$targetValImages = [Math]::Round($totalImages * 0.15)
$targetTestImages = $totalImages - $targetTrainImages - $targetValImages

$trainGroups = @()
$valGroups = @()
$testGroups = @()

$trainImageCount = 0
$valImageCount = 0
$testImageCount = 0

foreach ($grp in $groupArray) {
    $c = $grp.Count
    # Greedy allocation to closest match targets while preserving group integrity
    if (($trainImageCount + $c -le $targetTrainImages) -or ($valImageCount -ge $targetValImages -and $testImageCount -ge $targetTestImages)) {
        $trainGroups += $grp
        $trainImageCount += $c
    } elseif (($valImageCount + $c -le $targetValImages) -or ($testImageCount -ge $targetTestImages)) {
        $valGroups += $grp
        $valImageCount += $c
    } else {
        $testGroups += $grp
        $testImageCount += $c
    }
}

Write-Output "--- Partition Allocations ---"
Write-Output "Train: $($trainGroups.Count) acquisitions, $trainImageCount images ($([Math]::Round($trainImageCount/$totalImages*100, 2))%)"
Write-Output "Val:   $($valGroups.Count) acquisitions, $valImageCount images ($([Math]::Round($valImageCount/$totalImages*100, 2))%)"
Write-Output "Test:  $($testGroups.Count) acquisitions, $testImageCount images ($([Math]::Round($testImageCount/$totalImages*100, 2))%)"

# 4. Leakage Verification
$trainParents = New-Object 'System.Collections.Generic.HashSet[string]'
foreach ($g in $trainGroups) { [void]$trainParents.Add($g.Name) }

$valParents = New-Object 'System.Collections.Generic.HashSet[string]'
foreach ($g in $valGroups) { [void]$valParents.Add($g.Name) }

$testParents = New-Object 'System.Collections.Generic.HashSet[string]'
foreach ($g in $testGroups) { [void]$testParents.Add($g.Name) }

$leak_train_val = 0
foreach ($p in $trainParents) { if ($valParents.Contains($p)) { $leak_train_val++ } }

$leak_train_test = 0
foreach ($p in $trainParents) { if ($testParents.Contains($p)) { $leak_train_test++ } }

$leak_val_test = 0
foreach ($p in $valParents) { if ($testParents.Contains($p)) { $leak_val_test++ } }

Write-Output "--- Leakage Verification ---"
Write-Output "Train ∩ Val parent acquisitions: $leak_train_val"
Write-Output "Train ∩ Test parent acquisitions: $leak_train_test"
Write-Output "Val ∩ Test parent acquisitions: $leak_val_test"

$totalAssigned = $trainImageCount + $valImageCount + $testImageCount
Write-Output "Sum of split images: $totalAssigned / $totalImages"

if ($leak_train_val -eq 0 -and $leak_train_test -eq 0 -and $leak_val_test -eq 0 -and $totalAssigned -eq $totalImages) {
    $leakageStatus = "PASS"
} else {
    $leakageStatus = "FAIL"
}
Write-Output "SCENE-LEVEL LEAKAGE CHECK: $leakageStatus"

# 5. Export Split CSV Files
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

$trainRecords = foreach ($g in $trainGroups) {
    foreach ($item in $g.Group) {
        [PSCustomObject]@{
            filename = $item.filename
            parent_acquisition_id = $item.parent_acquisition_id
            split = "train"
            acquisition_start_time = $item.acquisition_start_time
            acquisition_stop_time = $item.acquisition_stop_time
            dataset_name = $item.dataset_name
            channels = $item.channels
            bands = $item.bands
            crs = $item.crs
            width = $item.width
            height = $item.height
        }
    }
}
$trainRecords | Export-Csv -Path (Join-Path $OutputDir "experiment2_train.csv") -NoTypeInformation -Encoding UTF8

$valRecords = foreach ($g in $valGroups) {
    foreach ($item in $g.Group) {
        [PSCustomObject]@{
            filename = $item.filename
            parent_acquisition_id = $item.parent_acquisition_id
            split = "val"
            acquisition_start_time = $item.acquisition_start_time
            acquisition_stop_time = $item.acquisition_stop_time
            dataset_name = $item.dataset_name
            channels = $item.channels
            bands = $item.bands
            crs = $item.crs
            width = $item.width
            height = $item.height
        }
    }
}
$valRecords | Export-Csv -Path (Join-Path $OutputDir "experiment2_val.csv") -NoTypeInformation -Encoding UTF8

$testRecords = foreach ($g in $testGroups) {
    foreach ($item in $g.Group) {
        [PSCustomObject]@{
            filename = $item.filename
            parent_acquisition_id = $item.parent_acquisition_id
            split = "test"
            acquisition_start_time = $item.acquisition_start_time
            acquisition_stop_time = $item.acquisition_stop_time
            dataset_name = $item.dataset_name
            channels = $item.channels
            bands = $item.bands
            crs = $item.crs
            width = $item.width
            height = $item.height
        }
    }
}
$testRecords | Export-Csv -Path (Join-Path $OutputDir "experiment2_test.csv") -NoTypeInformation -Encoding UTF8

Write-Output "Successfully wrote split CSVs to $OutputDir"
