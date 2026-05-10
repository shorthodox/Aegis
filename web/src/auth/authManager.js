export class AuthManager {
    static getToken() {
        return localStorage.getItem('access_token') || sessionStorage.getItem('access_token');
    }

    static setToken(token, remember = true) {
        if (remember) {
            localStorage.setItem('access_token', token);
        } else {
            sessionStorage.setItem('access_token', token);
        }
    }
    
    static clearToken() {
        localStorage.removeItem('access_token');
        sessionStorage.removeItem('access_token');
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
        if (userData && userData.trial && userData.trial.endDate) {
            // Support both Firestore Timestamp and ISO String
            let endDate = userData.trial.endDate;
            if (typeof endDate === 'object' && endDate.seconds) {
                endDate = new Date(endDate.seconds * 1000).toISOString();
            }
            localStorage.setItem('trial_end_timestamp', endDate);
        }
    }

    static getUser() {
        const data = localStorage.getItem('user_profile');
        return data ? JSON.parse(data) : null;
    }

    static getPlanType() {
        const user = this.getUser();
        if (!user) return 'free_trial';
        return user.plan || 'free_trial';
    }

    static isTrialValid() {
        const trialEnd = localStorage.getItem('trial_end_timestamp');
        if (!trialEnd) return false;
        return new Date() < new Date(trialEnd);
    }

    static isTokenValid() {
        const token = this.getToken();
        if (!token) return false;
        try {
            const payload = JSON.parse(atob(token.split('.')[1]));
            // Add a 5 minute buffer for token expiration
            return payload.exp * 1000 > Date.now() + 300000;
        } catch (e) {
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
            const payload = JSON.parse(atob(token.split('.')[1]));
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
        if (!data) return false;
        
        let hasBaseAccess = false;
        if (data.plan_type === 'active' || data.plan_type === 'pro') {
            hasBaseAccess = true;
        } else if (data.plan_type === 'free_trial') {
            if (data.trial_start) {
                const trialStartMs = (typeof data.trial_start === 'number' && data.trial_start < 10000000000) 
                    ? data.trial_start * 1000 
                    : new Date(data.trial_start).getTime();
                
                const hoursSinceStart = (Date.now() - trialStartMs) / (1000 * 60 * 60);
                if (hoursSinceStart < 24) {
                    hasBaseAccess = true;
                }
            }
        }
        
        if (!hasBaseAccess) return false;
        
        if (feature === 'signals') return true;
        if (feature === 'extended_timeframes') {
            return data.plan_type === 'active' || data.plan_type === 'pro';
        }
        
        return false;
    }

    static getSubscriptionStatus() {
        const user = this.getUser();
        if (!user) return 'expired';
        
        // If pro, they have full access
        if (user.plan === 'pro') return 'active';
        
        // Use new token-based logic if available
        if (this.getUserData()?.plan_type) {
             return this.hasAccess('signals') ? 'active' : 'expired';
        }
        
        // Fallback to local storage validity
        if (this.isTrialValid()) {
            return 'active';
        }
        
        return 'expired';
    }
}
