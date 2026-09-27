param(
    [string]$ManifestCsv = "archive_images_list_utf8_parsed.csv",
    [string]$ExtractedDir = "extracted_dataset\images"
)

$manifestData = Import-Csv -Path $ManifestCsv
$manifestItems = @($manifestData | Where-Object { $_.FullName -match '\.tif$' })

$extractedFiles = @(Get-ChildItem -Path $ExtractedDir -Filter "*.tif" -Recurse)
$extractedNames = @($extractedFiles | ForEach-Object { $_.Name })

$extractedSet = New-Object 'System.Collections.Generic.HashSet[string]' (,[string[]]$extractedNames)

$missing = @()
foreach ($m in $manifestItems) {
    $leaf = Split-Path $m.FullName -Leaf
    if (-not $extractedSet.Contains($leaf)) {
        $missing += $leaf
    }
}

$duplicateGroups = @($extractedNames | Group-Object | Where-Object { $_.Count -gt 1 })

[PSCustomObject]@{
    ExpectedImages = $manifestItems.Count
    ExtractedImages = $extractedFiles.Count
    MissingImages = $missing.Count
    DuplicateFilenames = $duplicateGroups.Count
} | Format-List
