# run this script to append large files to the .gitignore in the current directory
# this will not overwrite previous contents of the .gitignore file but will not add a file if it already exists
# and will remove files that no longer exist on disk

$gitignore_path = (Get-ChildItem "./.gitignore").FullName
Write-Host $gitignore_path

Get-Content -Path $gitignore_path

$files = Get-ChildItem . -Recurse

foreach ($file in $files){
    if($file.Length -gt 500KB){
        Write-Host $file
    }
}
