export class AuthManager {
    static getToken() {
        return localStorage.getItem('access_token') ||
               localStorage.getItem('authToken') ||
               sessionStorage.getItem('access_token') ||
               sessionStorage.getItem('authToken');
    }

    static setToken(token, remember = true) {
        if (remember) {
            localStorage.setItem('access_token', token);
            localStorage.setItem('authToken', token);
        } else {
            sessionStorage.setItem('access_token', token);
            sessionStorage.setItem('authToken', token);
        }
    }
    
    static clearToken() {
        localStorage.removeItem('access_token');
        localStorage.removeItem('authToken');
        sessionStorage.removeItem('access_token');
        sessionStorage.removeItem('authToken');
    }
    
    static isAuthenticated() {
        return !!this.getToken();
    }
    
    static getAuthHeader() {
        const token = this.getToken();
        return token ? `Bearer ${token}` : null;
    }

    static setUser(userData) {
        localStorage.setItem('user_profile', JSON.stringify(userData));
        if (userData) {
            let endDate = null;
            if (userData.trial && userData.trial.endDate !== undefined && userData.trial.endDate !== null) {
                endDate = userData.trial.endDate;
            } else if (userData.trial_end !== undefined && userData.trial_end !== null) {
                endDate = userData.trial_end;
            } else if (userData.trialEnd !== undefined && userData.trialEnd !== null) {
                endDate = userData.trialEnd;
            }
            
            if (endDate && String(endDate) !== 'null') {
                // Support both Firestore Timestamp and ISO String
                if (typeof endDate === 'object' && endDate.seconds) {
                    endDate = new Date(endDate.seconds * 1000).toISOString();
                } else if (typeof endDate === 'number') { // Support ms timestamp
                    endDate = new Date(endDate).toISOString();
                }
                localStorage.setItem('trial_end_timestamp', endDate);
                console.log('[AuthManager] trial_end_timestamp updated:', endDate);
            } else {
                console.warn('[AuthManager] No valid trial end date found in userData:', userData);
            }
        }
    }

    static getUser() {
        const data = localStorage.getItem('user_profile');
        return data ? JSON.parse(data) : null;
    }

    static getPlanType() {
        const tokenData = this.getUserData();
        if (tokenData && tokenData.plan_type) {
            return tokenData.plan_type;
        }
        
        const user = this.getUser();
        if (user && user.plan) {
            return user.plan;
        }
        
        return 'free_trial';
    }

    static isTrialValid() {
        // Retrieve the trial_end_timestamp from localStorage.
        const trialEndStr = localStorage.getItem('trial_end_timestamp');
        if (trialEndStr && trialEndStr !== 'null') {
            const trialEnd = new Date(trialEndStr).getTime();
            if (Date.now() > trialEnd) {
                return false;
            }
        }

        // Exclusively use server response data
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
        
        // Strict security posture: Default to inactive/expired state instead of active
        return false;
    }

    static isTokenValid() {
        const token = this.getToken();
        if (!token) return false;
        try {
            const base64Url = token.split('.')[1];
            const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
            const payload = JSON.parse(atob(base64));
            // Add a small 10 second buffer for token expiration instead of 5 minutes
            return payload.exp * 1000 > Date.now() + 10000;
        } catch (e) {
            console.error('Token validation failed:', e);
            return false;
        }
    }

    static async refreshToken() {
        try {
            const response = await fetch('/api/auth/refresh', {
                method: 'POST',
                headers: {
                    'Authorization': this.getAuthHeader()
                }
            });
            if (response.ok) {
                const data = await response.json();
                if (data.access_token) {
                    this.setToken(data.access_token);
                    return true;
                }
            }
            return false;
        } catch (error) {
            console.error('Token refresh failed', error);
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
                plan_type: payload.plan_type,
                status: payload.status
            };
        } catch (e) {
            console.error('Failed to decode token payload', e);
            return null;
        }
    }

    static hasAccess(feature) {
        const data = this.getUserData();
        const user = this.getUser();
        
        // Check both Token and Local User Profile
        const plan = data?.plan_type || user?.plan || user?.plan_type;
        const status = data?.status || user?.status;

        if (plan === 'active' || plan === 'pro' || status === 'active') {
            return true;
        }
        
        // Fallback for trials
        if (plan === 'free_trial' || plan === 'trial') {
            return this.isTrialValid();
        }
        
        return false;
    }

    static getSubscriptionStatus() {
        const user = this.getUser();
        
        // Immediately return active if the user has a pro/active plan
        if (user && (user.plan === 'pro' || user.plan === 'active')) {
            return 'active';
        }

        // Prioritize actual trial end date stored in the user profile/localStorage
        if (this.isTrialValid()) {
            return 'active';
        }

        if (user && (user.plan === 'trial' || user.plan === 'free_trial') && user.trial_expired === false) {
            return 'active';
        }
        
        // Fallback to decoded JWT logic
        const tokenData = this.getUserData();
        if (tokenData) {
            return this.hasAccess('signals') ? 'active' : 'expired';
        }
        
        return 'expired';
    }

    /**
     * Helper method to check if user is fully authenticated and has valid token
     * Returns a Promise for backward compatibility with async code
     * @returns {Promise<boolean>}
     */
    static async checkAuth() {
        const isAuth = this.isAuthenticated();
        const isTokenOk = this.isTokenValid();
        return isAuth && isTokenOk;
    }
}

// Make AuthManager globally available for non-module scripts
if (typeof window !== 'undefined') {
    window.AuthManager = AuthManager;
}
