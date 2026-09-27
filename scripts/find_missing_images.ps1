param(
    [string]$ManifestCsv = "archive_images_list_utf8_parsed.csv",
    [string]$ExtractedDir = "extracted_dataset\images",
    [string]$OutputFile = "missing_images.csv"
)

$manifestData = Import-Csv -Path $ManifestCsv
$manifestItems = @($manifestData | Where-Object { $_.FullName -match '\.tif$' })

$extractedFiles = @(Get-ChildItem -Path $ExtractedDir -Filter "*.tif" -Recurse | ForEach-Object { $_.Name })
$extractedSet = New-Object 'System.Collections.Generic.HashSet[string]' (,[string[]]$extractedFiles)

$missingList = [System.Collections.Generic.List[PSCustomObject]]::new()
foreach ($m in $manifestItems) {
    $leaf = Split-Path $m.FullName -Leaf
    if (-not $extractedSet.Contains($leaf)) {
        $missingList.Add([PSCustomObject]@{
            Filename = $leaf
            ArchivePath = $m.FullName
            Folder = $m.Folder
            BaseName = $m.BaseName
            Size = $m.Size
        })
    }
}

Write-Host "Authoritative manifest TIFF count: $($manifestItems.Count)"
Write-Host "Extracted dataset TIFF count: $($extractedFiles.Count)"
Write-Host "Total missing TIFF count: $($missingList.Count)"

$missingList | Export-Csv -Path $OutputFile -NoTypeInformation -Encoding UTF8
Write-Host "Saved missing image list to $OutputFile"

if ($missingList.Count -gt 0) {
    Write-Host "`nFirst 10 missing files:"
    $missingList | Select-Object -First 10 | Format-Table -AutoSize | Out-String | Write-Host
    Write-Host "Last 10 missing files:"
    $missingList | Select-Object -Last 10 | Format-Table -AutoSize | Out-String | Write-Host
}
