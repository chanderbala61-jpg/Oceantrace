[CmdletBinding()]
param(
    [string]$SourceDir = "C:\Users\bala\OneDrive\Documents\exp 1\Oceantrace_Colab_Exp2",
    [string]$UploadPartsDir = "C:\Users\bala\OneDrive\Documents\exp 1\Oceantrace_Colab_Exp2_UploadParts",
    [switch]$SampleHashOnly
)

$ErrorActionPreference = "Stop"

Write-Host "=================================================="
Write-Host "COLAB PACKAGE 10-PART INTEGRITY VERIFICATION"
Write-Host "Source: $SourceDir"
Write-Host "UploadParts: $UploadPartsDir"
Write-Host "=================================================="

# 1. Check Manifest
$manifestFile = Join-Path $UploadPartsDir "UPLOAD_MANIFEST.csv"
if (-not (Test-Path $manifestFile)) {
    throw "Manifest file not found: $manifestFile"
}

$manifestRows = Import-Csv $manifestFile
Write-Host "Manifest entries loaded: $($manifestRows.Count)"

# 2. Check 10 Part Directories
$expectedParts = 1..10 | ForEach-Object { "Part_{0:D2}" -f $_ }
$partsFound = Get-ChildItem -LiteralPath $UploadPartsDir -Directory | Where-Object { $_.Name -like "Part_*" } | Sort-Object Name

Write-Host "Found $($partsFound.Count) part directories."
if ($partsFound.Count -ne 10) {
    throw "Expected exactly 10 part directories! Found: $($partsFound.Count)"
}

# 3. Count Original Files
$origFiles = Get-ChildItem -LiteralPath $SourceDir -Recurse -File
$origTotalCount = $origFiles.Count
$origTotalBytes = [long]($origFiles | Measure-Object -Property Length -Sum).Sum
Write-Host "`nOriginal Package:"
Write-Host "  Total Files: $origTotalCount"
Write-Host "  Total Bytes: $origTotalBytes ($([math]::Round($origTotalBytes / 1GB, 3)) GB)"

# 4. Count Files across all 10 Parts
$partFiles = @()
$partStats = @{}

foreach ($p in $expectedParts) {
    $pDir = Join-Path $UploadPartsDir $p
    if (-not (Test-Path $pDir)) {
        throw "Part directory missing: $pDir"
    }
    $pFiles = Get-ChildItem -LiteralPath $pDir -Recurse -File
    $pSize = [long]($pFiles | Measure-Object -Property Length -Sum).Sum
    $partStats[$p] = @{
        FileCount = $pFiles.Count
        SizeBytes = $pSize
        SizeGB = [math]::Round($pSize / 1GB, 3)
    }
    $partFiles += $pFiles
}

$partsTotalCount = $partFiles.Count
$partsTotalBytes = [long]($partFiles | Measure-Object -Property Length -Sum).Sum

Write-Host "`nUpload Parts Total:"
Write-Host "  Total Files: $partsTotalCount"
Write-Host "  Total Bytes: $partsTotalBytes ($([math]::Round($partsTotalBytes / 1GB, 3)) GB)"

# 5. Part Size and File Count Breakdown
Write-Host "`nPart Breakdown:"
foreach ($p in $expectedParts) {
    Write-Host ("  {0}: {1} files, {2} GB ({3} bytes)" -f $p, $partStats[$p].FileCount, $partStats[$p].SizeGB, $partStats[$p].SizeBytes)
}

# 6. Verify File Counts Match
$countMatch = ($origTotalCount -eq $partsTotalCount) -and ($partsTotalCount -eq $manifestRows.Count)
Write-Host "`nVerification Checks:"
Write-Host "1. File Count Match: $countMatch (Orig: $origTotalCount, Parts: $partsTotalCount, Manifest: $($manifestRows.Count))"

# 7. Verify Every File Appears Exactly Once (Zero duplicates, Zero missing)
$relPathMap = @{}
$duplicates = [System.Collections.Generic.List[string]]::new()

foreach ($pf in $partFiles) {
    # Extract relative path inside its part
    $partName = $pf.FullName.Substring($UploadPartsDir.Length).TrimStart('\', '/').Split([char[]]@('\', '/'))[0]
    $rel = $pf.FullName.Substring((Join-Path $UploadPartsDir $partName).Length).TrimStart('\', '/').Replace('\', '/')
    
    if ($relPathMap.ContainsKey($rel)) {
        $duplicates.Add($rel)
    } else {
        $relPathMap[$rel] = $partName
    }
}

$missingFromParts = [System.Collections.Generic.List[string]]::new()
foreach ($of in $origFiles) {
    $rel = $of.FullName.Substring($SourceDir.Length).TrimStart('\', '/').Replace('\', '/')
    if (-not $relPathMap.ContainsKey($rel)) {
        $missingFromParts.Add($rel)
    }
}

Write-Host "2. Unique Files (no duplicates): $($duplicates.Count -eq 0) (Duplicates found: $($duplicates.Count))"
Write-Host "3. Completeness (no missing): $($missingFromParts.Count -eq 0) (Missing files: $($missingFromParts.Count))"

# 8. Byte Size Verification
$bytesMatch = ($origTotalBytes -eq $partsTotalBytes)
Write-Host "4. Total Bytes Match: $bytesMatch (Difference: $($partsTotalBytes - $origTotalBytes) bytes)"

# 9. Verify 1200 Images and 1200 Masks
$imagesCount = ($relPathMap.Keys | Where-Object { $_ -like "extracted_dataset/images/*" }).Count
$masksCount = ($relPathMap.Keys | Where-Object { $_ -like "extracted_dataset/masks/*" }).Count
Write-Host "5. 1,200 Images Present: $($imagesCount -eq 1200) (Count: $imagesCount)"
Write-Host "6. 1,200 Masks Present: $($masksCount -eq 1200) (Count: $masksCount)"

# 10. Verify Mandatory Experiment 2 Files
$mandatoryFiles = @(
    "data/splits/experiment2_train.csv",
    "data/splits/experiment2_val.csv",
    "data/splits/experiment2_test.csv",
    "configs/exp2_segformer_b0.yaml",
    "src/models_exp2.py",
    "src/train_exp2.py",
    "src/datasets/experiment2_sar_dataset.py",
    "requirements_exp2_colab.txt"
)

$mandatoryPresent = $true
Write-Host "7. Mandatory Code/Config Files Check:"
foreach ($mf in $mandatoryFiles) {
    $present = $relPathMap.ContainsKey($mf)
    if (-not $present) {
        $mandatoryPresent = $false
        Write-Host "   [MISSING] $mf" -ForegroundColor Red
    } else {
        Write-Host "   [OK] $mf (in $($relPathMap[$mf]))"
    }
}

# 11. SHA-256 Hash Verification Against Manifest
Write-Host "`n8. Verifying Manifest SHA-256 Hashes against disk files (SampleHashOnly: $SampleHashOnly)..."
$hashErrors = [System.Collections.Generic.List[string]]::new()

$partSampleCounts = @{}
$checkedCount = 0
$hashedCount = 0

foreach ($row in $manifestRows) {
    $relWin = $row.relative_path.Replace('/', '\')
    $filePath = [System.IO.Path]::Combine($UploadPartsDir, $row.part, $relWin)
    
    if (-not (Test-Path $filePath)) {
        $hashErrors.Add("File missing on disk: $filePath")
        continue
    }
    
    $fInfo = New-Object System.IO.FileInfo($filePath)
    if ($fInfo.Length -ne [long]$row.file_size) {
        $hashErrors.Add("Size mismatch for $($row.relative_path): expected $($row.file_size), got $($fInfo.Length)")
        continue
    }
    
    $checkedCount++

    # Determine whether to compute SHA-256
    $shouldHash = $true
    if ($SampleHashOnly) {
        $isSupporting = ($row.relative_path -notlike "extracted_dataset/*")
        if (-not $isSupporting) {
            $cat = if ($row.relative_path -like "*images*") { "img" } else { "mask" }
            $sampleKey = "$($row.part)_$cat"
            if (-not $partSampleCounts.ContainsKey($sampleKey)) {
                $partSampleCounts[$sampleKey] = 0
            }
            if ($partSampleCounts[$sampleKey] -ge 5) {
                $shouldHash = $false
            } else {
                $partSampleCounts[$sampleKey]++
            }
        }
    }

    if ($shouldHash) {
        $sha = [System.Security.Cryptography.SHA256]::Create()
        $inStream = [System.IO.File]::OpenRead($filePath)
        $hashBytes = $sha.ComputeHash($inStream)
        $inStream.Close()
        $sha.Dispose()
        
        $computedHash = [BitConverter]::ToString($hashBytes).Replace('-', '').ToLower()
        if ($computedHash -ne $row.sha256.ToLower()) {
            $hashErrors.Add("Hash mismatch for $($row.relative_path): expected $($row.sha256), got $computedHash")
        }
        $hashedCount++
    }
    
    if ($checkedCount % 400 -eq 0) {
        Write-Host "   Checked: $checkedCount / $($manifestRows.Count) files (Hashed: $hashedCount)..."
    }
}

Write-Host "   Checked size & path for all $checkedCount files."
Write-Host "   Computed and verified SHA-256 hashes for $hashedCount files."
$hashCheckPass = ($hashErrors.Count -eq 0)
Write-Host "   SHA-256 Hash Integrity Result: $hashCheckPass (Errors: $($hashErrors.Count))"

# 12. Final Certification Decision
$allPassed = $countMatch -and ($duplicates.Count -eq 0) -and ($missingFromParts.Count -eq 0) -and $bytesMatch -and ($imagesCount -eq 1200) -and ($masksCount -eq 1200) -and $mandatoryPresent -and $hashCheckPass

Write-Host "`n=================================================="
if ($allPassed) {
    Write-Host "VERIFICATION RESULT: PASS" -ForegroundColor Green
    Write-Host "UPLOAD PARTS AUDIT: PASS" -ForegroundColor Green
} else {
    Write-Host "VERIFICATION RESULT: FAIL" -ForegroundColor Red
}
Write-Host "=================================================="
