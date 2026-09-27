param([string]$listPath)

$lines = Get-Content $listPath
$files = @()
$datePattern = "^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}"

foreach ($line in $lines) {
    if ($line -match $datePattern) {
        $parts = -split $line
        # format: Date Time Attr Size Compressed Name
        # or Date Time Attr Size Name
        $name = $parts[-1]
        $size = $parts[-2]
        if ($name -match "\.tif$") {
            $base = [System.IO.Path]::GetFileNameWithoutExtension($name)
            $folder = [System.IO.Path]::GetDirectoryName($name)
            $files += [PSCustomObject]@{
                FullName = $name
                Folder = $folder
                BaseName = $base
                Size = $size
            }
        }
    }
}
$files | Export-Csv -Path ($listPath -replace "\.txt$", "_parsed.csv") -NoTypeInformation
Write-Output "Parsed $($files.Count) tif entries from $listPath"
