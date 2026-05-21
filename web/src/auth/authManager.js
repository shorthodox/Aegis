export class AuthManager {
    // ── Token Storage ────────────────────────────────────────────────
    // Canonical key: 'access_token'. 'authToken' is kept as a read-only
    // legacy fallback so old sessions survive across deploys.

    static getToken() {
        return localStorage.getItem('access_token') ||
               localStorage.getItem('authToken') ||
               sessionStorage.getItem('access_token') ||
               sessionStorage.getItem('authToken');
    }

    static setToken(token, remember = true) {
        if (remember) {
            localStorage.setItem('access_token', token);
            localStorage.removeItem('authToken'); // retire legacy duplicate
        } else {
            sessionStorage.setItem('access_token', token);
            sessionStorage.removeItem('authToken');
        }
    }

    static clearToken() {
        localStorage.removeItem('access_token');
        localStorage.removeItem('authToken');
        sessionStorage.removeItem('access_token');
        sessionStorage.removeItem('authToken');
    }

    // ── User Profile Storage ─────────────────────────────────────────

    static clearUser() {
        localStorage.removeItem('user_profile');
        localStorage.removeItem('trial_end_timestamp');
        localStorage.removeItem('trial_end_sig');
        localStorage.removeItem('cachedAllowedTokens');
    }

    // Full logout: wipe tokens + user data in one call.
    static logout() {
        this.clearToken();
        this.clearUser();
    }

    static isAuthenticated() {
        return !!this.getToken();
    }

    static getAuthHeader() {
        const token = this.getToken();
        return token ? `Bearer ${token}` : null;
    }

    // ── Trial End Integrity ──────────────────────────────────────────
    // SHA-256 hash of `uid||trialEnd||salt` stored alongside the
    // trial_end_timestamp. Prevents casual DevTools manipulation.
    // Note: the WebSocket server-authority tick (~1 s) remains the
    // primary enforcement mechanism; this is a defense-in-depth layer.

    static _signTrialEnd(trialEndIso, uid) {
        if (typeof crypto === 'undefined' || !crypto.subtle) return;
        const payload = `${uid}||${trialEndIso}||aegis-integrity-v1`;
        crypto.subtle
            .digest('SHA-256', new TextEncoder().encode(payload))
            .then(buf => {
                const hex = Array.from(new Uint8Array(buf))
                    .map(b => b.toString(16).padStart(2, '0'))
                    .join('');
                localStorage.setItem('trial_end_sig', hex);
            })
            .catch(() => {});
    }

    // Returns true (valid), false (tampered), or null (cannot verify).
    static async verifyTrialEndIntegrity() {
        const trialEndIso = localStorage.getItem('trial_end_timestamp');
        const storedSig   = localStorage.getItem('trial_end_sig');
        if (!trialEndIso || !storedSig) return null;

        const user = this.getUser();
        if (!user) return null;

        const uid = user.uid || user.email || '';
        if (typeof crypto === 'undefined' || !crypto.subtle) return null;

        try {
            const payload = `${uid}||${trialEndIso}||aegis-integrity-v1`;
            const buf = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(payload));
            const expected = Array.from(new Uint8Array(buf))
                .map(b => b.toString(16).padStart(2, '0'))
                .join('');
            return storedSig === expected;
        } catch {
            return null;
        }
    }

    // ── User Profile ─────────────────────────────────────────────────

    static setUser(userData) {
        localStorage.setItem('user_profile', JSON.stringify(userData));
        if (!userData) return;

        let endDate = null;
        if (userData.trial?.endDate != null) {
            endDate = userData.trial.endDate;
        } else if (userData.trial_end != null) {
            endDate = userData.trial_end;
        } else if (userData.trialEnd != null) {
            endDate = userData.trialEnd;
        }

        if (endDate && String(endDate) !== 'null') {
            if (typeof endDate === 'object' && endDate.seconds) {
                endDate = new Date(endDate.seconds * 1000).toISOString();
            } else if (typeof endDate === 'number') {
                endDate = new Date(endDate).toISOString();
            }
            localStorage.setItem('trial_end_timestamp', endDate);
            // Sign for tamper detection
            const uid = userData.uid || userData.email || '';
            this._signTrialEnd(endDate, uid);
        }
    }

    static getUser() {
        const data = localStorage.getItem('user_profile');
        if (!data) return null;
        try {
            return JSON.parse(data);
        } catch {
            // Corrupted entry — remove it so the app does not get stuck.
            localStorage.removeItem('user_profile');
            return null;
        }
    }

    // ── Plan & Trial ─────────────────────────────────────────────────

    static getPlanType() {
        const tokenData = this.getUserData();
        if (tokenData?.plan_type) return tokenData.plan_type;

        const user = this.getUser();
        if (user?.plan) return user.plan;

        return 'free_trial';
    }

    static isTrialValid() {
        const trialEndStr = localStorage.getItem('trial_end_timestamp');
        if (trialEndStr && trialEndStr !== 'null') {
            const trialEnd = new Date(trialEndStr).getTime();
            if (isNaN(trialEnd)) {
                // Unparseable value — clear it to prevent perpetual loop.
                localStorage.removeItem('trial_end_timestamp');
                return false;
            }
            return Date.now() <= trialEnd;
        }

        const user = this.getUser();
        if (user) {
            if (user.plan === 'pro' || user.plan === 'active' || user.subscription_active) return true;
            if (user.trial_expired === false) return true;
            if (user.trial_expired === true) return false;
        }

        const tokenData = this.getUserData();
        if (tokenData) {
            if (tokenData.plan_type === 'pro' || tokenData.plan_type === 'active' || tokenData.status === 'active') return true;
            if (tokenData.status === 'expired') return false;
        }

        // Fail-secure: assume expired when no authoritative data is present.
        return false;
    }

    static isTokenValid() {
        const token = this.getToken();
        if (!token) return false;
        try {
            const base64Url = token.split('.')[1];
            const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
            const payload = JSON.parse(atob(base64));
            return payload.exp * 1000 > Date.now() + 10000; // 10-second buffer
        } catch {
            return false;
        }
    }

    static async refreshToken() {
        try {
            const response = await fetch('/api/auth/refresh', {
                method: 'POST',
                headers: { 'Authorization': this.getAuthHeader() }
            });
            if (response.ok) {
                const data = await response.json();
                if (data.access_token) {
                    this.setToken(data.access_token);
                    return true;
                }
            }
            return false;
        } catch {
            return false;
        }
    }

    static getUserData() {
        const token = this.getToken();
        if (!token) return null;
        try {
            const base64Url = token.split('.')[1];
            const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
            const payload = JSON.parse(atob(base64));
            return {
                trial_start: payload.trial_start,
                plan_type:   payload.plan_type,
                status:      payload.status
            };
        } catch {
            return null;
        }
    }

    static hasAccess(_feature) {
        const data = this.getUserData();
        const user = this.getUser();

        const plan   = data?.plan_type || user?.plan || user?.plan_type;
        const status = data?.status    || user?.status;

        if (plan === 'active' || plan === 'pro' || status === 'active') return true;
        if (plan === 'free_trial' || plan === 'trial') return this.isTrialValid();

        return false;
    }

    static getSubscriptionStatus() {
        const user = this.getUser();
        if (user && (user.plan === 'pro' || user.plan === 'active')) return 'active';
        if (this.isTrialValid()) return 'active';
        if (user && (user.plan === 'trial' || user.plan === 'free_trial') && user.trial_expired === false) return 'active';

        const tokenData = this.getUserData();
        if (tokenData) return this.hasAccess('signals') ? 'active' : 'expired';

        return 'expired';
    }

    static async checkAuth() {
        return this.isAuthenticated() && this.isTokenValid();
    }
}

// Make AuthManager globally available for non-module scripts.
if (typeof window !== 'undefined') {
    window['AuthManager'] = AuthManager;
}
