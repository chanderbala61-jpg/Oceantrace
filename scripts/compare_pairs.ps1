$imgCsv = Import-Csv "archive_images_list_utf8_parsed.csv"
$maskCsv = Import-Csv "archive_masks_list_utf8_parsed.csv"

Write-Output "=== FILE COUNT CHECK ==="
Write-Output "Image TIFF count: $($imgCsv.Count)"
Write-Output "Mask TIFF count:  $($maskCsv.Count)"

$imgBases = [System.Collections.Generic.HashSet[string]]::new([string[]]($imgCsv.BaseName))
$maskBases = [System.Collections.Generic.HashSet[string]]::new([string[]]($maskCsv.BaseName))

$missingMasks = @()
foreach ($b in $imgBases) {
    if (-not $maskBases.Contains($b)) {
        $missingMasks += $b
    }
}

$missingImages = @()
foreach ($b in $maskBases) {
    if (-not $imgBases.Contains($b)) {
        $missingImages += $b
    }
}

Write-Output "Images missing masks: $($missingMasks.Count)"
if ($missingMasks.Count -gt 0) {
    Write-Output "Examples: $($missingMasks[0..4] -join ', ')"
}

Write-Output "Masks missing images: $($missingImages.Count)"
if ($missingImages.Count -gt 0) {
    Write-Output "Examples: $($missingImages[0..4] -join ', ')"
}

# Check duplicate names within images
$imgDupes = $imgCsv | Group-Object BaseName | Where-Object { $_.Count -gt 1 }
Write-Output "Duplicate Image BaseNames: $($imgDupes.Count)"

# Check duplicate names within masks
$maskDupes = $maskCsv | Group-Object BaseName | Where-Object { $_.Count -gt 1 }
Write-Output "Duplicate Mask BaseNames: $($maskDupes.Count)"

# Mask file sizes
$maskSizes = $maskCsv | Group-Object Size
Write-Output "`n=== MASK FILE SIZES IN ARCHIVE ==="
foreach ($g in $maskSizes) {
    Write-Output "Size $($g.Name) bytes : count $($g.Count)"
}

# Image file sizes summary
$imgSizes = $imgCsv | Measure-Object -Property Size -Minimum -Maximum -Average -Sum
Write-Output "`n=== IMAGE UNCOMPRESSED FILE SIZES ==="
Write-Output "Min: $($imgSizes.Minimum) bytes"
Write-Output "Max: $($imgSizes.Maximum) bytes"
Write-Output "Avg: $([math]::Round($imgSizes.Average)) bytes"
Write-Output "Sum: $($imgSizes.Sum) bytes ($([math]::Round($imgSizes.Sum / 1GB, 2)) GB)"
