[CmdletBinding()]
param(
    [string]$SourceDir = "C:\Users\bala\OneDrive\Documents\exp 1\Oceantrace_Colab_Exp2",
    [string]$TargetDir = "C:\Users\bala\OneDrive\Documents\exp 1\Oceantrace_Colab_Exp2_UploadParts"
)

$ErrorActionPreference = "Stop"

Write-Host "=================================================="
Write-Host "COLAB PACKAGE 10-PART SPLITTER & STREAM HASHER"
Write-Host "Source: $SourceDir"
Write-Host "Target: $TargetDir"
Write-Host "=================================================="

# 1. Ensure Target Directory exists
if (-not (Test-Path $TargetDir)) {
    New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
}

# 2. Gather All Source Files
Write-Host "Scanning source files in $SourceDir..."
$allSourceFiles = Get-ChildItem -LiteralPath $SourceDir -Recurse -File

$totalSourceCount = $allSourceFiles.Count
$totalSourceBytes = ($allSourceFiles | Measure-Object -Property Length -Sum).Sum

Write-Host "Found $totalSourceCount files, total $([math]::Round($totalSourceBytes / 1GB, 3)) GB ($totalSourceBytes bytes)."

# 3. Categorize files: Images, Masks, and Other (Code, Configs, Docs)
$imgFiles = @($allSourceFiles | Where-Object { $_.FullName -match [regex]::Escape("\extracted_dataset\images\") } | Sort-Object Name)
$maskFiles = @($allSourceFiles | Where-Object { $_.FullName -match [regex]::Escape("\extracted_dataset\masks\") } | Sort-Object Name)
$otherFiles = @($allSourceFiles | Where-Object { $_.FullName -notmatch [regex]::Escape("\extracted_dataset\images\") -and $_.FullName -notmatch [regex]::Escape("\extracted_dataset\masks\") } | Sort-Object FullName)

Write-Host "Images count: $($imgFiles.Count)"
Write-Host "Masks count:  $($maskFiles.Count)"
Write-Host "Other count:  $($otherFiles.Count)"

if ($imgFiles.Count -ne 1200 -or $maskFiles.Count -ne 1200) {
    throw "Expected 1200 images and 1200 masks! Aborting."
}

# 4. Map each file to its assigned part
# Part_01 gets 120 images, 120 masks, plus all 21 other files + empty output folders
# Part_02..10 get 120 images, 120 masks each
$assignments = [System.Collections.Generic.List[PSCustomObject]]::new()

for ($partIdx = 0; $partIdx -lt 10; $partIdx++) {
    $partNum = $partIdx + 1
    $partName = ("Part_{0:D2}" -f $partNum)
    $startIdx = $partIdx * 120
    $endIdx = $startIdx + 119
    
    # Assign images
    for ($i = $startIdx; $i -le $endIdx; $i++) {
        $assignments.Add([PSCustomObject]@{
            FileObj = $imgFiles[$i]
            Part = $partName
        })
    }
    
    # Assign masks
    for ($i = $startIdx; $i -le $endIdx; $i++) {
        $assignments.Add([PSCustomObject]@{
            FileObj = $maskFiles[$i]
            Part = $partName
        })
    }
    
    # If Part_01, assign other files
    if ($partNum -eq 1) {
        foreach ($of in $otherFiles) {
            $assignments.Add([PSCustomObject]@{
                FileObj = $of
                Part = $partName
            })
        }
    }
}

Write-Host "Total file assignments: $($assignments.Count) (Must match source: $totalSourceCount)"
if ($assignments.Count -ne $totalSourceCount) {
    throw "Assignment count mismatch! Aborting."
}

# 5. Prepare Manifest & Load Existing Records for Instant Resume
$manifestPath = Join-Path $TargetDir "UPLOAD_MANIFEST.csv"
$existingMap = @{}
if (Test-Path $manifestPath) {
    try {
        $existingRows = Import-Csv $manifestPath
        foreach ($r in $existingRows) {
            $key = "$($r.part):$($r.relative_path)"
            $existingMap[$key] = @{
                sha256 = $r.sha256
                size = [long]$r.file_size
            }
        }
        Write-Host "Loaded $($existingMap.Count) existing entries from $manifestPath for instant resume."
    } catch {
        Write-Host "Could not read existing manifest for resume, will re-verify files."
    }
}

$manifestEntries = [System.Collections.Generic.List[string]]::new()
$manifestEntries.Add("relative_path,filename,file_size,sha256,part")

# 6. Prepare Buffers and Crypto Provider
$bufferSize = 4 * 1024 * 1024 # 4 MB
$buffer = New-Object byte[] $bufferSize

$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$bytesCopied = [long]0
$filesCopied = 0

$currentPart = ""

foreach ($item in $assignments) {
    $f = $item.FileObj
    $part = $item.Part
    
    if ($part -ne $currentPart) {
        if ($currentPart -ne "") {
            $elapsedSec = [math]::Max(1, [int]($stopwatch.Elapsed.TotalSeconds))
            $mbps = [math]::Round(($bytesCopied / 1MB) / $elapsedSec, 1)
            Write-Host "Completed $currentPart. Total Copied: $([math]::Round($bytesCopied / 1GB, 2)) GB | Speed: $mbps MB/s"
        }
        $currentPart = $part
        Write-Host "`n>>> Processing $part..."
        
        # Ensure part root directory exists
        $partRoot = Join-Path $TargetDir $part
        [System.IO.Directory]::CreateDirectory($partRoot) | Out-Null
        
        # If Part_01, also create the empty output directories
        if ($part -eq "Part_01") {
            $outDirs = @(
                (Join-Path $partRoot "outputs\experiment2_segformer_b0\checkpoints"),
                (Join-Path $partRoot "outputs\experiment2_segformer_b0\figures"),
                (Join-Path $partRoot "outputs\experiment2_segformer_b0\logs")
            )
            foreach ($od in $outDirs) {
                [System.IO.Directory]::CreateDirectory($od) | Out-Null
            }
        }
    }
    
    # Calculate relative path
    $relPathWin = $f.FullName.Substring($SourceDir.Length).TrimStart('\', '/')
    $relPathPosix = $relPathWin.Replace('\', '/')
    
    $destFilePath = [System.IO.Path]::Combine($TargetDir, $part, $relPathWin)
    $destFileDir = [System.IO.Path]::GetDirectoryName($destFilePath)
    if (-not [System.IO.Directory]::Exists($destFileDir)) {
        [System.IO.Directory]::CreateDirectory($destFileDir) | Out-Null
    }
    
    # Check if destination file already exists with identical size (Resume support)
    $alreadyExists = $false
    if ([System.IO.File]::Exists($destFilePath)) {
        $destInfo = New-Object System.IO.FileInfo($destFilePath)
        if ($destInfo.Length -eq $f.Length) {
            $alreadyExists = $true
        }
    }

    $key = "$($part):$($relPathPosix)"
    $hashStr = ""

    if ($alreadyExists -and $existingMap.ContainsKey($key) -and ($existingMap[$key].size -eq $f.Length)) {
        # Instant resume: file exists with correct size and already hashed in manifest!
        $hashStr = $existingMap[$key].sha256
    } elseif ($alreadyExists) {
        # File already copied, stream hash existing destination file
        $sha = [System.Security.Cryptography.SHA256]::Create()
        $inStream = New-Object System.IO.FileStream($destFilePath, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read, $bufferSize)
        $bytesRead = 0
        while (($bytesRead = $inStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
            $sha.TransformBlock($buffer, 0, $bytesRead, $null, 0) | Out-Null
        }
        $sha.TransformFinalBlock($buffer, 0, 0) | Out-Null
        $inStream.Close()
        $hashStr = [BitConverter]::ToString($sha.Hash).Replace('-', '').ToLower()
    } else {
        # Stream copy + SHA256
        $sha = [System.Security.Cryptography.SHA256]::Create()
        $inStream = New-Object System.IO.FileStream($f.FullName, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read, $bufferSize)
        $outStream = New-Object System.IO.FileStream($destFilePath, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None, $bufferSize)
        
        $bytesRead = 0
        while (($bytesRead = $inStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
            $sha.TransformBlock($buffer, 0, $bytesRead, $null, 0) | Out-Null
            $outStream.Write($buffer, 0, $bytesRead)
        }
        $sha.TransformFinalBlock($buffer, 0, 0) | Out-Null
        
        $inStream.Close()
        $outStream.Close()
        
        # Preserve original timestamps
        [System.IO.File]::SetCreationTimeUtc($destFilePath, $f.CreationTimeUtc)
        [System.IO.File]::SetLastWriteTimeUtc($destFilePath, $f.LastWriteTimeUtc)
        $hashStr = [BitConverter]::ToString($sha.Hash).Replace('-', '').ToLower()
    }
    
    $fileSize = [long]$f.Length
    
    $bytesCopied += $fileSize
    $filesCopied++
    
    # Add to manifest
    $manifestEntries.Add("$relPathPosix,$($f.Name),$fileSize,$hashStr,$part")
    
    # Flush manifest periodically (every 50 files) for crash resiliency
    if ($filesCopied % 50 -eq 0) {
        [System.IO.File]::WriteAllLines($manifestPath, $manifestEntries)
    }

    if ($filesCopied % 200 -eq 0) {
        $elapsedSec = [math]::Max(1, [int]($stopwatch.Elapsed.TotalSeconds))
        $mbps = [math]::Round(($bytesCopied / 1MB) / $elapsedSec, 1)
        Write-Host "Progress: $filesCopied / $totalSourceCount files ($([math]::Round($bytesCopied / 1GB, 2)) / $([math]::Round($totalSourceBytes / 1GB, 2)) GB) - $mbps MB/s"
    }
}

$stopwatch.Stop()
Write-Host "`nAll files copied and hashed in $($stopwatch.Elapsed.ToString('hh\:mm\:ss'))!"

# Final Manifest Write
Write-Host "Finalizing UPLOAD_MANIFEST.csv..."
[System.IO.File]::WriteAllLines($manifestPath, $manifestEntries)
Write-Host "Manifest successfully saved to $manifestPath with $($manifestEntries.Count - 1) entries."
