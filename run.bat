@echo off
setlocal enabledelayedexpansion
set gofiles=
for %%f in (*.go) do (
    set gofiles=!gofiles! %%f
)
go run !gofiles! %*