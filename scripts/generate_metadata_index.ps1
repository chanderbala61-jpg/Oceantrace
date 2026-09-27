param(
    [string]$ImagesDir = "extracted_dataset\images",
    [string]$OutputFile = "metadata_index.csv"
)

$files = Get-ChildItem -Path $ImagesDir -Filter "*.tif" | Sort-Object Name
Write-Output "Processing $($files.Count) TIFF files from $ImagesDir..."

$results = @()
$counter = 0

foreach ($file in $files) {
    $counter++
    if ($counter % 50 -eq 0 -or $counter -eq $files.Count) {
        Write-Output "Processed $counter / $($files.Count) files..."
    }

    $stream = [System.IO.File]::OpenRead($file.FullName)
    $buffer = New-Object byte[] 65536
    $bytesRead = $stream.Read($buffer, 0, 65536)
    
    # Read first IFD
    $firstIfd = ([uint32]$buffer[4] -shl 24) -bor ([uint32]$buffer[5] -shl 16) -bor ([uint32]$buffer[6] -shl 8) -bor [uint32]$buffer[7]
    $numEntries = ([uint16]$buffer[$firstIfd] -shl 8) -bor [uint16]$buffer[$firstIfd+1]
    
    $tag65000_offset = 0
    $tag65000_count = 0
    $width = 0
    $length = 0
    $samplesPerPixel = 0
    
    for ($i = 0; $i -lt $numEntries; $i++) {
        $entryOffset = $firstIfd + 2 + ($i * 12)
        $tag = ([uint16]$buffer[$entryOffset] -shl 8) -bor [uint16]$buffer[$entryOffset+1]
        $valOffset = $entryOffset + 8
        
        if ($tag -eq 256) { $width = ([uint16]$buffer[$valOffset] -shl 8) -bor [uint16]$buffer[$valOffset+1] }
        if ($tag -eq 257) { $length = ([uint16]$buffer[$valOffset] -shl 8) -bor [uint16]$buffer[$valOffset+1] }
        if ($tag -eq 277) { $samplesPerPixel = ([uint16]$buffer[$valOffset] -shl 8) -bor [uint16]$buffer[$valOffset+1] }
        if ($tag -eq 65000) {
            $tag65000_count = ([uint32]$buffer[$entryOffset+4] -shl 24) -bor ([uint32]$buffer[$entryOffset+5] -shl 16) -bor ([uint32]$buffer[$entryOffset+6] -shl 8) -bor [uint32]$buffer[$entryOffset+7]
            $tag65000_offset = ([uint32]$buffer[$valOffset] -shl 24) -bor ([uint32]$buffer[$valOffset+1] -shl 16) -bor ([uint32]$buffer[$valOffset+2] -shl 8) -bor [uint32]$buffer[$valOffset+3]
        }
    }
    
    # Read first 100KB of Tag 65000 (XML) which contains all high-level scene attributes
    $xmlBytesLen = [Math]::Min(102400, $tag65000_count)
    $xmlBuffer = New-Object byte[] $xmlBytesLen
    $stream.Seek($tag65000_offset, [System.IO.SeekOrigin]::Begin) | Out-Null
    $stream.Read($xmlBuffer, 0, $xmlBytesLen) | Out-Null
    $stream.Close()
    
    $xmlStr = [System.Text.Encoding]::UTF8.GetString($xmlBuffer)
    
    # Extract metadata attributes
    $datasetName = if ($xmlStr -match '<DATASET_NAME>(.*?)</DATASET_NAME>') { $matches[1] } else { "" }
    $startTime = if ($xmlStr -match '<PRODUCT_SCENE_RASTER_START_TIME>(.*?)</PRODUCT_SCENE_RASTER_START_TIME>') { $matches[1] } else { "" }
    $stopTime = if ($xmlStr -match '<PRODUCT_SCENE_RASTER_STOP_TIME>(.*?)</PRODUCT_SCENE_RASTER_STOP_TIME>') { $matches[1] } else { "" }
    $orbitPass = if ($xmlStr -match '<PASS desc=".*?">(.*?)</PASS>') { $matches[1] } else { "" }
    $absOrbit = if ($xmlStr -match '<ABS_ORBIT desc=".*?">(.*?)</ABS_ORBIT>') { $matches[1] } else { "" }
    $relOrbit = if ($xmlStr -match '<REL_ORBIT desc=".*?">(.*?)</REL_ORBIT>') { $matches[1] } else { "" }
    $latRes = if ($xmlStr -match '<lat_pixel_res desc=".*?">(.*?)</lat_pixel_res>') { $matches[1] } else { "" }
    $lonRes = if ($xmlStr -match '<lon_pixel_res desc=".*?">(.*?)</lon_pixel_res>') { $matches[1] } else { "" }
    $nearLat = if ($xmlStr -match '<first_near_lat.*?>(.*?)</first_near_lat>') { $matches[1] } else { "" }
    $nearLong = if ($xmlStr -match '<first_near_long.*?>(.*?)</first_near_long>') { $matches[1] } else { "" }
    $farLat = if ($xmlStr -match '<last_far_lat.*?>(.*?)</last_far_lat>') { $matches[1] } else { "" }
    $farLong = if ($xmlStr -match '<last_far_long.*?>(.*?)</last_far_long>') { $matches[1] } else { "" }
    $offsetX = if ($xmlStr -match '<subset_offset_x.*?>(.*?)</subset_offset_x>') { $matches[1] } else { "" }
    $offsetY = if ($xmlStr -match '<subset_offset_y.*?>(.*?)</subset_offset_y>') { $matches[1] } else { "" }
    
    # Derive Parent Sentinel-1 Product ID from datasetName
    # Example datasetName: subset_0_of_S1A_IW_GRDH_1SDV_20180803T172551_20180803T172608_023085_0281B1_DB30_Orb_NR_Cal_Spk_TC_dB
    $parentProduct = ""
    if ($datasetName -match 'of_(S1[AB]_.*)$') {
        $parentProduct = $matches[1]
    } else {
        $parentProduct = $datasetName
    }

    $results += [PSCustomObject]@{
        filename = $file.Name
        parent_acquisition_id = $parentProduct
        dataset_name = $datasetName
        acquisition_start_time = $startTime
        acquisition_stop_time = $stopTime
        orbit_pass = $orbitPass
        absolute_orbit = $absOrbit
        relative_orbit_track = $relOrbit
        crs = "EPSG:4326 (WGS 84)"
        width = $width
        height = $length
        channels = $samplesPerPixel
        bands = "Sigma0_VH_db, Sigma0_VV_db"
        lat_pixel_res_deg = $latRes
        lon_pixel_res_deg = $lonRes
        bbox_first_near_lat = $nearLat
        bbox_first_near_long = $nearLong
        bbox_last_far_lat = $farLat

        
        bbox_last_far_long = $farLong
        subset_offset_x = $offsetX
        subset_offset_y = $offsetY
    }
}

$results | Export-Csv -Path $OutputFile -NoTypeInformation -Encoding UTF8
Write-Output "Successfully saved $OutputFile with $($results.Count) records."
