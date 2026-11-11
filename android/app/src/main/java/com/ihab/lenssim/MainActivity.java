package com.ihab.lenssim;

import android.app.Activity;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.Manifest;
import android.content.pm.PackageManager;
import android.webkit.PermissionRequest;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import androidx.core.content.FileProvider;
import java.io.File;

/** Full-screen WebView around the IHAB Lens Thickness Simulator (served from GitHub Pages, so it updates by itself). */
public class MainActivity extends Activity {
    private static final String HOME = "https://ihabali300-png.github.io/Ihab-thickness-simulator/";
    private static final int PICK = 41;
    private WebView web;
    private ValueCallback<Uri[]> pending;
    private Uri cameraUri;
    private PermissionRequest pendingPerm;
    private static final int CAM_PERM = 42;

    @Override protected void onCreate(Bundle saved) {
        super.onCreate(saved);
        web = new WebView(this);
        setContentView(web);
        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        s.setDatabaseEnabled(true);
        s.setMediaPlaybackRequiresUserGesture(false);
        s.setCacheMode(WebSettings.LOAD_DEFAULT);
        s.setUserAgentString(s.getUserAgentString() + " " + BuildConfig.UA_TAG);
        web.setWebViewClient(new WebViewClient() {
            @Override public boolean shouldOverrideUrlLoading(WebView v, WebResourceRequest r) {
                Uri u = r.getUrl();
                if ("ihabali300-png.github.io".equals(u.getHost())) return false;
                try { startActivity(new Intent(Intent.ACTION_VIEW, u)); } catch (ActivityNotFoundException e) { }
                return true;
            }
        });
        web.setWebChromeClient(new WebChromeClient() {
            @Override public void onPermissionRequest(final PermissionRequest req) {
                runOnUiThread(() -> {
                    boolean wantsCam = false;
                    for (String r : req.getResources()) if (PermissionRequest.RESOURCE_VIDEO_CAPTURE.equals(r)) wantsCam = true;
                    if (!wantsCam) { req.deny(); return; }
                    if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                        req.grant(new String[]{ PermissionRequest.RESOURCE_VIDEO_CAPTURE });
                    } else { pendingPerm = req; requestPermissions(new String[]{ Manifest.permission.CAMERA }, CAM_PERM); }
                });
            }
            @Override public boolean onShowFileChooser(WebView v, ValueCallback<Uri[]> cb, FileChooserParams p) {
                if (pending != null) pending.onReceiveValue(null);
                pending = cb;
                Intent intent;
                if (p.isCaptureEnabled()) {          // <input capture> → open the camera directly
                    try {
                        File dir = new File(getCacheDir(), "photos"); dir.mkdirs();
                        File f = new File(dir, "shot_" + System.currentTimeMillis() + ".jpg");
                        cameraUri = FileProvider.getUriForFile(MainActivity.this, getPackageName() + ".files", f);
                        intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                        intent.putExtra(MediaStore.EXTRA_OUTPUT, cameraUri);
                        intent.addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION | Intent.FLAG_GRANT_READ_URI_PERMISSION);
                    } catch (Exception e) { cameraUri = null; intent = p.createIntent(); }
                } else {                              // gallery / files (images, PDF, anything the page accepts)
                    cameraUri = null;
                    intent = new Intent(Intent.ACTION_GET_CONTENT);
                    intent.addCategory(Intent.CATEGORY_OPENABLE);
                    java.util.ArrayList<String> types = new java.util.ArrayList<>();
                    boolean any = false;
                    for (String t : p.getAcceptTypes()) {
                        for (String x : t.split(",")) { x = x.trim(); if (x.isEmpty()) continue; if (x.equals("*/*")) any = true; else if (x.contains("/")) types.add(x); else if (x.equals(".pdf")) types.add("application/pdf"); }
                    }
                    if (any) intent.setType("*/*");
                    else if (types.isEmpty()) intent.setType("image/*");
                    else if (types.size() == 1) intent.setType(types.get(0));
                    else { intent.setType("*/*"); intent.putExtra(Intent.EXTRA_MIME_TYPES, types.toArray(new String[0])); }
                    if (p.getMode() == FileChooserParams.MODE_OPEN_MULTIPLE) intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
                }
                try { startActivityForResult(intent, PICK); }
                catch (ActivityNotFoundException e) { pending.onReceiveValue(null); pending = null; return false; }
                return true;
            }
        });
        if (saved != null) web.restoreState(saved); else web.loadUrl(HOME + BuildConfig.START);
    }

    @Override protected void onActivityResult(int req, int res, Intent data) {
        if (req != PICK || pending == null) { super.onActivityResult(req, res, data); return; }
        Uri[] out = null;
        if (res == RESULT_OK) {
            if (cameraUri != null) out = new Uri[]{ cameraUri };
            else if (data != null && data.getClipData() != null) {
                int n = data.getClipData().getItemCount(); out = new Uri[n];
                for (int i = 0; i < n; i++) out[i] = data.getClipData().getItemAt(i).getUri();
            }
            else if (data != null && data.getData() != null) out = new Uri[]{ data.getData() };
        }
        pending.onReceiveValue(out); pending = null;
    }

    @Override public void onRequestPermissionsResult(int code, String[] perms, int[] res) {
        if (code == CAM_PERM && pendingPerm != null) {
            if (res.length > 0 && res[0] == PackageManager.PERMISSION_GRANTED) pendingPerm.grant(new String[]{ PermissionRequest.RESOURCE_VIDEO_CAPTURE });
            else pendingPerm.deny();
            pendingPerm = null;
        }
    }

    @Override protected void onSaveInstanceState(Bundle out) { super.onSaveInstanceState(out); web.saveState(out); }

    @Override public void onBackPressed() { if (web.canGoBack()) web.goBack(); else super.onBackPressed(); }
}
