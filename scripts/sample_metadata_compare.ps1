$names = @("00000.tif", "00002.tif", "00642.tif")

foreach ($name in $names) {
    $filePath = "temp_audit_samples\images\$name"
    $bytes = [System.IO.File]::ReadAllBytes($filePath)
    
    $str = [System.Text.Encoding]::UTF8.GetString($bytes, 0, [Math]::Min(100000, $bytes.Length))
    
    Write-Output "=================================================="
    Write-Output "FILE: $name (Size: $($bytes.Length) bytes)"
    Write-Output "=================================================="
    
    # Check dataset name / product name
    if ($str -match '<DATASET_NAME>(.*?)</DATASET_NAME>') {
        Write-Output "Product Name: $($matches[1])"
    }
    if ($str -match '<PRODUCT_SCENE_RASTER_START_TIME>(.*?)</PRODUCT_SCENE_RASTER_START_TIME>') {
        Write-Output "Acquisition Start: $($matches[1])"
    }
    if ($str -match '<PRODUCT_SCENE_RASTER_STOP_TIME>(.*?)</PRODUCT_SCENE_RASTER_STOP_TIME>') {
        Write-Output "Acquisition Stop:  $($matches[1])"
    }
    if ($str -match '<PASS desc=".*?">(.*?)</PASS>') {
        Write-Output "Orbit Pass: $($matches[1])"
    }
    if ($str -match '<ABS_ORBIT desc=".*?">(.*?)</ABS_ORBIT>') {
        Write-Output "Absolute Orbit: $($matches[1])"
    }
    if ($str -match '<REL_ORBIT desc=".*?">(.*?)</REL_ORBIT>') {
        Write-Output "Relative Orbit (Track): $($matches[1])"
    }
    if ($str -match '<first_near_lat.*?>(.*?)</first_near_lat>') {
        Write-Output "Lat Bounds: $($matches[1])"
    }
    if ($str -match '<first_near_long.*?>(.*?)</first_near_long>') {
        Write-Output "Lon Bounds: $($matches[1])"
    }
    if ($str -match '<subset_offset_x.*?>(.*?)</subset_offset_x>') {
        Write-Output "Parent Scene Offset X: $($matches[1])"
    }
    if ($str -match '<subset_offset_y.*?>(.*?)</subset_offset_y>') {
        Write-Output "Parent Scene Offset Y: $($matches[1])"
    }
}
